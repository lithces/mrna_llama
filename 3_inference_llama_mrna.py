# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""
import tqdm
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union
from typing import Any, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from lit_gpt.utils import CycleIterator, chunked_cross_entropy, num_parameters



#%%
######### DATASET SETUP, please test the loader before proceed
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset

class TokenTensorDataset(LocalDataset):
    '''
    padding with -1, later will be substituted in train function 
    - with vocab_size for input
    - remains -1 for target
    '''
    def __init__(self, localpath, ctx_size):
        super().__init__(local=localpath)
        self.ctx_size = ctx_size+1 # need to add one for AR nature.


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        L = len(dat)

        if L < self.ctx_size:            
            padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
            return torch.concat( (dat.to(torch.int16), padding))
        else:
            return dat[:self.ctx_size]


def create_dataloaders(batch_size, block_size):
    # Remote directory (S3 or local filesystem) where dataset is stored
    ds_val = TokenTensorDataset(in_data_path, ctx_size=block_size)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return dl_val
#########
#%%
from mrna_utils import *
mrna_tok = MRNATOK()

# System settings
model_name = "tiny-llama-1.1b"
name = "lit-tiny-llama-1.1b"

out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
logger_name = "tensorboard"
devices = torch.cuda.device_count() or 1

# Hyperparameters
############
#### modify me please
# global_batch_size = 512

micro_batch_size = 12*4 # modify me please
global_batch_size = micro_batch_size*devices

rope_condense_ratio = 2
ctx_size = 8192
# n_embd = 2048
# intermediate_size = 5632
n_embd = 1024 # dont change me
intermediate_size = 5632//2

log_step_interval = 100
eval_iters = 1000
save_step_interval = 20000
eval_step_interval = 20000
max_tokens = ctx_size*10e6*2  

############

learning_rate = 4e-4
warmup_steps = 2000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

# in_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds/va"
# out_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_mamba/va"

# in_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds/va"
# out_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_llama/va"

in_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds/va"
out_data_path = f"/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_llama_rc_{rope_condense_ratio}/va"
os.makedirs(f"/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_llama_rc_{rope_condense_ratio}", exist_ok=True)



batch_size = global_batch_size // devices
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
warmup_iters = warmup_steps * gradient_accumulation_iters
log_iter_interval = log_step_interval * gradient_accumulation_iters


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


class MyGPT(GPT):
    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return x  # (b, t, vocab_size)
    



def setup(resume: Union[bool, Path] = False):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(fabric, resume)



def main(fabric, resume):
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    ############## modify the config for mRNA cases
    config = Config.from_name(model_name)
    ############## modify the config for mRNA cases
    config.block_size = ctx_size
    config.vocab_size = len(mrna_tok.map_id_tok)
    config.padded_vocab_size = len(mrna_tok.map_id_tok)+1
    config.n_embd = n_embd
    config.intermediate_size = intermediate_size
    config.hf_config['name'] = 'mrna_tinyllama'
    config.hf_config['org'] = 'Sanofi'
    config.__post_init__()
    config.rope_condense_ratio = rope_condense_ratio
    print(mrna_tok.map_id_tok)
    #######
    #######
    print(config)
    val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = MyGPT(config)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model, 
        "optimizer": optimizer, 
        "hparams": hparams, 
        "iter_num": 0, 
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1].split('.')[0])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    compute_embeddings(fabric, model, val_dataloader)

@torch.no_grad()
def compute_embeddings(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int=9999999) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    columns = {'embs':'pkl', 'L': 'int', 'loss': 'float32'}
    shutil.rmtree(out_data_path, ignore_errors=True)
    with MDSWriter(out=out_data_path, columns=columns, compression=None, keep_local=True) as out:
        for k, val_data in tqdm.tqdm(enumerate(val_dataloader)):
            if k >= max_iters:
                break

            input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
            targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
            idx_ignore_batch = input_ids<0
            input_ids[idx_ignore_batch] = model.config.vocab_size

            M = input_ids.shape[0]
            hidden = model.forward(input_ids)
            logits = model.lm_head(hidden)
            lst_loss = [torch.nn.functional.cross_entropy(logits[mi], targets[mi], ignore_index=-1, size_average=True, reduce=True).item() for mi in range(M)]
            lst_L = idx_ignore_batch.long().argmax(dim=1).detach().cpu().numpy().tolist()
            
            hidden = hidden.cpu().numpy().astype(np.float16)


            # print('###### ', lst_L.shape, lst_L.dtype)
            idx_ignore_batch = idx_ignore_batch.cpu().numpy()

            for mi in range(M):
                towrite = hidden[mi, ~(idx_ignore_batch[mi]),:]
                sample = {
                    'embs': towrite,
                    'L': lst_L[mi],
                    'loss': lst_loss[mi]
                }                
                out.write(sample)


def choose_logger(logger_name: str, name: str, resume: Union[bool, Path], *args, **kwargs):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", *args, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", *args, **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project="tinyllama", name=name, resume=(resume is not False), *args, **kwargs)
    raise ValueError(f"`logger={logger_name}` is not a valid option.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI
    from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2

    if not _TORCH_GREATER_EQUAL_2_2:
        raise ImportError("The tinyllama.py training script requires PyTorch 2.2 (nightly) or higher to run.")

    CLI(setup)

