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


#%%
######### DATASET SETUP, please test the loader before proceed
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
import random
# class TokenTensorDataset(LocalDataset):
#     def __init__(self, local, ctx_size):
#         super().__init__(local=local)
#         self.ctx_size = ctx_size+1 # need to add one for AR nature.


#     def __getitem__(self, index: int):
#         obj = super().__getitem__(index)
#         dat = torch.tensor(obj['ids'])
#         L = len(dat)
#         th = np.random.rand()
#         if th<0.5:
#             if L < self.ctx_size:            
#                 padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
#                 return torch.concat( (dat.to(torch.int16), padding))
#             else:
#                 return dat[:self.ctx_size]
#         else:
#             if L < self.ctx_size:            
#                 padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
#                 return torch.concat( (padding, dat.to(torch.int16)))
#             else:
#                 return dat[-self.ctx_size:]

class TokenTensorDataset(LocalDataset):
    '''
    padding with -1, later will be substituted in train function 
    - with vocab_size for input
    - remains -1 for target
    '''
    def __init__(self, local, ctx_size):
        super().__init__(local=local)
        self.ctx_size = ctx_size+1 # need to add one for AR nature.


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        L = len(dat)
        th = np.random.rand()
        if L < self.ctx_size:
            padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
            return torch.concat( (dat.to(torch.int16), padding))
        else:
            # randomly pickup a starting point
            diff_L = L - self.ctx_size
            st = random.randint(0, diff_L)
            return dat[st:(st+self.ctx_size)]



def create_dataloaders(batch_size, block_size):
    # Remote directory (S3 or local filesystem) where dataset is stored

    # Local directory where dataset is cached during operation
    lst_cate = [
        "rabbits"
        ,"monotremes"
        ,"insectivores"
        ,"odd-toed_ungulates"
        ,"more_placentals"
        ,"bats"
        ,"even-toed_ungulates"
        ,"primates"
        ,"carnivores"
        ,"rodents"
        ,"marsupials"
    ]
    ds_root = '/data2/data/mrna_llm/raw_seqs_cds_pos/np_after_tok_mds_th20000/'




    local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
    lst_ds = [TokenTensorDataset(local=li, ctx_size=ctx_size) for li in local_dirs]
    lst_rngs = [torch.Generator().manual_seed(42) for catei in lst_cate]
    lst_splits = [random_split(ds, [0.95, 0.05], generator=rng) for ds, rng in zip(lst_ds, lst_rngs)]
    lst_ds_train = [si[0] for si in lst_splits]
    lst_ds_val = [si[1] for si in lst_splits]

    ds_train = ConcatDataset(lst_ds_train)
    ds_val = ConcatDataset(lst_ds_val)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
    return dl_train, dl_val
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
global_batch_size = 12
micro_batch_size = 12 # modify me please
ctx_size = 1024
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


def main(fabric, resume):
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

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
    print(mrna_tok.map_id_tok)
    #######
    print(config)
    train_dataloader, val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = GPT(config)
        model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))

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
        "train_dataloader": train_dataloader,
        "hparams": hparams, 
        "iter_num": 0, 
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1].split('.')[0])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = max_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0:model.config.block_size].contiguous().long()
        input_ids[input_ids<0] = model.config.vocab_size

        targets = train_data[:, 1:(model.config.block_size + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        # print('###  1')
        running_loss.update(loss.detach())
        # print('###  2')

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        # print('###  3')

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * micro_batch_size),
                lengths=(state["iter_num"] * micro_batch_size * model.config.block_size),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * micro_batch_size * model.config.block_size,
                "total_tokens": state["iter_num"] * micro_batch_size * model.config.block_size * fabric.world_size,
                "learning_rate": lr,
            }

            fabric.print(
                f"iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, iter time:"
                f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step),' if not is_accumulating else ','}"
                f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"])
        # print('###  4')

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()
        # print('###  5')

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"step-{state['step_count']:08d}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
        # print('###  6')
        # print('### is_accu', is_accumulating)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(max_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
        targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
        input_ids[input_ids<0] = model.config.vocab_size
        # print('##########', input_ids.shape, input_ids.min(), input_ids.max())

        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses[k] = loss

    model.train()
    return losses.mean()

# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def init_weights(module: nn.Module, n_layer: int, n_embd: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        if name == "proj.weight" and isinstance(module, (LLaMAMLP, CausalSelfAttention)):
            nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(n_embd) / n_layer))


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


#%%
1+1
# %%
