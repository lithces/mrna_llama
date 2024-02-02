#%%

from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
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
]
ds_root = '/data2/data/raw_seqs_cds_pos/np_after_tok_mds/'



class TokenTensorDataset(LocalDataset):
    def __init__(self, local, ctx_size=2048):
        super().__init__(local=local)
        self.ctx_size = ctx_size


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        L = len(dat)
        if L < self.ctx_size:            
            padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
            return torch.concat( (dat.to(torch.int16), padding))
        else:
            return dat[:self.ctx_size]

local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
lst_ds = [TokenTensorDataset(local=li, ctx_size=2048) for li in local_dirs]
lst_rngs = [torch.Generator().manual_seed(42) for catei in lst_cate]
lst_splits = [random_split(ds, [0.95, 0.05], generator=rng) for ds, rng in zip(lst_ds, lst_rngs)]
lst_ds_train = [si[0] for si in lst_splits]
lst_ds_val = [si[1] for si in lst_splits]

ds_train = ConcatDataset(lst_ds_train)
ds_val = ConcatDataset(lst_ds_val)

dl_train = DataLoader(ds_train, batch_size=8, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=8, shuffle=False)

# %%
# cds = TokenTensorDataset(local="/data2/data/raw_seqs_cds_pos/np_after_tok_mds/rabbits", ctx_size=2048)
# dl_debug = DataLoader(cds, batch_size=8, shuffle=True)
dl_debug = dl_train
import tqdm
cnt = 0
for xi in tqdm.tqdm(dl_debug):
    # cnt +=1
    # if cnt >=10:
    #     break
    cnt+=xi.shape[0]
from matplotlib import pyplot as plt
plt.plot(xi.t())

# %%
if 0:
    import pandas as pd
    df = pd.read_parquet('/data2/data/raw_seqs_cds_pos/rabbits.parquet.snappy')
# %%
# %%
sum([len(dsi) for dsi in lst_ds])
# %%
