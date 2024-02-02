#%%
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
from lightning.data import StreamingDataset
from lightning.data.streaming.item_loader import TokensLoader
from lightning.data.streaming.dataloader import StreamingDataLoader
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from lit_gpt.utils import CycleIterator, chunked_cross_entropy, num_parameters

# System settings
devices = torch.cuda.device_count() or 1



block_size = 10000
# Increase by one because we need the next word as well
effective_block_size = block_size + 1

train_datasets = \
    StreamingDataset(
        input_dir="/data2/data/raw_seqs_cds_pos/tensor_after_tok/rabbits",
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=False,
        drop_last=True,
    )


N = len(train_datasets)
print('N', N)
train_dataloader = DataLoader(
    train_datasets, batch_size=10000, pin_memory=True, num_workers=1, drop_last=False
)


for xi in train_dataloader:
    yy = xi.contiguous().long()
    print(yy.shape)
    break

xxx
#%%
xi = xi.to(torch.uint8)
#%%
import numpy as np
unique_values, counts = torch.unique(xi.view(-1,), return_counts=True)

# %%
counts
# %%
