#%%
import numpy as np
from streaming import MDSWriter

import pandas as pd
import gc
import glob
import os
import shutil
#%% first, extract stats to distributed parquet files, free memory
import ctypes
def free_mem():
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)


# Local or remote directory in which to store the compressed output files
data_in = '/data2/data/raw_seqs_cds_pos/'



lst_fn = glob.glob(f'{data_in}/*.parquet.snappy')


from collections import defaultdict, Counter
stat_e = defaultdict(lambda: 0)


#%%

#%%
import tqdm

for fi in lst_fn:
    basename = os.path.basename(fi)
    mytype = basename.split('.')[0]
    print(mytype)        

    free_mem()

    df = pd.read_parquet(fi)
    for dati in tqdm.tqdm(df.itertuples()):
        seq, st, ed = dati[1], dati[2], dati[3]
        if ( (ed-st)%3 !=0) or ( (ed-st)>=10000):
            continue
        reti = dict(Counter(seq))
        for ki, vi in reti.items():
            stat_e[ki] += vi
        # sample = {
        #     'ids': cret,
        #     'class': mytype,
        # }
        # out.write(sample)
        # break
    
    del df
    free_mem()

# %%
stat_e