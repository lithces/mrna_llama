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
ds_root = '/data2/data/mrna_llm/raw_seqs_cds_pos/np_after_tok_mds/'


import numpy as np
class TokenTensorDataset(LocalDataset):
    def __init__(self, local, ctx_size=2048):
        super().__init__(local=local)
        self.ctx_size = ctx_size


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        L = len(dat)
        th = np.random.rand()
        if th<0.5:
            if L < self.ctx_size:            
                padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
                return torch.concat( (dat.to(torch.int16), padding))
            else:
                return dat[:self.ctx_size]
        else:
            if L < self.ctx_size:            
                padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
                return torch.concat( (padding, dat.to(torch.int16)))
            else:
                return dat[-self.ctx_size:]

local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
lst_ds = [TokenTensorDataset(local=li, ctx_size=2048) for li in local_dirs]
lst_rngs = [torch.Generator().manual_seed(42) for catei in lst_cate]
lst_splits = [random_split(ds, [0.95, 0.05], generator=rng) for ds, rng in zip(lst_ds, lst_rngs)]
lst_ds_train = [si[0] for si in lst_splits]
lst_ds_val = [si[1] for si in lst_splits]

ds_train = ConcatDataset(lst_ds_train)
ds_val = ConcatDataset(lst_ds_val)

dl_train = DataLoader(ds_train, batch_size=8, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=128, shuffle=False)

# %%
import tqdm
for dati in tqdm.tqdm(dl_val):
    print(dati)
    break
# %%
print((dati[:,-1]==3).sum())
print((dati[:,0]==0).sum())

print((dati[:,0]==0).sum())
print((dati[:,-1]==-1).sum())

# %%
