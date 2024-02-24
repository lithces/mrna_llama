# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""

import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

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

import shutil
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import *
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

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
# model_name = "tiny-llama-1.1b"
name = "mamba"
out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
logger_name = "tensorboard"
devices = torch.cuda.device_count() or 1

# Hyperparameters
############
#### modify me please
# global_batch_size = 512
global_batch_size = 12
micro_batch_size = 12 # modify me please
ctx_size = 8192

# d_model = 2560
# n_layer = 64
d_model = 512
n_layer = 12


log_step_interval = 100
eval_iters = 1000
save_step_interval = 20000
eval_step_interval = 20000
max_tokens = ctx_size*10e6*2  

############

learning_rate = 1e-4
warmup_steps = 2000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

# in_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds/va"
# out_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_mamba/va"

in_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds/tr"
out_data_path = "/home/lithtp/sanofi/mrna/mrna_downstream/mds_emb_mamba/tr"


batch_size = global_batch_size // devices
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
warmup_iters = warmup_steps * gradient_accumulation_iters
log_iter_interval = log_step_interval * gradient_accumulation_iters


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(resume: Union[bool, Path] = False):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(fabric, resume)

class MyMamba(MambaLMHeadModel):
    
    # def __init__(
    #     self,
    #     config: MambaConfig,
    #     initializer_cfg=None,
    #     device=None,
    #     dtype=None,
    # ) -> None:
    #     print('#$$$$$$$$$$$$$$$$$')
    #     super().__init__(config, initializer_cfg, device, dtype)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone.forward(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        return hidden_states


def main(fabric, resume):w
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    ############## modify the config for mRNA cases
    config = MambaConfig(d_model=d_model, n_layer=n_layer, vocab_size=len(mrna_tok.map_id_tok)+1)
    config.block_size = ctx_size
    print(mrna_tok.map_id_tok)
    #######
    print(config)
    val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = MyMamba(config)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # model = torch.compile(model)
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
    columns = {'embs':'pkl'}
    shutil.rmtree(out_data_path, ignore_errors=True)
    with MDSWriter(out=out_data_path, columns=columns, compression=None, keep_local=True) as out:
        for k, val_data in enumerate(val_dataloader):
            if k >= max_iters:
                break
            input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()

            idx_ignore_all = input_ids<0
            input_ids[idx_ignore_all] = model.config.vocab_size

            hidden = model.forward(input_ids).cpu().numpy().astype(np.float16)
            idx_ignore_all = idx_ignore_all.cpu().numpy()
            
            M = input_ids.shape[0]

            for mi in range(M):
                towrite = hidden[mi, ~(idx_ignore_all[mi]),:]
                sample = {
                    'embs': towrite,
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

