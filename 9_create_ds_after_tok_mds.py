#%%
import numpy as np
from streaming import MDSWriter

import pandas as pd
import gc
import glob
import os
import shutil
import re
#%% first, extract stats to distributed parquet files, free memory
import ctypes
def free_mem():
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

# A dictionary mapping input fields to their data types
columns = {
    'ids': 'pkl',
}

#%%
import numpy as np
from mrna_utils import *

mrnatok = MRNATOK()
#%%
# test code below
if 0:
    tseq = 'dfAUAUGATCUACUGAfgshfAUC'
    tst, ted = 4, len(tseq)-8
    # mrnatok.validate_mrna(tseq, tst, ted)
    ret = mrnatok.get_ids(tseq, tst, ted)
    decode_ret = [mrnatok.map_id_tok[xi] for xi in ret]
    print(tseq)
    print(decode_ret)
#%%
#%%
import tqdm

# %%
if 0:
    from pathlib import Path
    from lit_gpt import Tokenizer
    tokenizer = Tokenizer(Path('/home/lithtp/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/./snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc'))
    text = 'this is a big cat'
    text_ids = tokenizer.encode(text, bos=False, eos=True)
    print(text_ids)
# %%
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
# %%
def myrun(
    input_dir_root: str = "/data2/data/raw_seqs_cds_pos/",
    output_dir_root: str = "/data2/data/raw_seqs_cds_pos/np_after_tok_mds/",
    mrna_cate: str = "rabbit",
) -> None:
    fi = f'{input_dir_root}/{mrna_cate}.parquet.snappy'
    mytype = mrna_cate
    print('---- loading', mytype)        

    free_mem()

    df = pd.read_parquet(fi)
    print(len(df))
    print('done loading, tokenizing')
    data_out_path = f'{output_dir_root}/{mytype}/'
    shutil.rmtree(data_out_path, ignore_errors=True)
    print(data_out_path)
    with MDSWriter(out=data_out_path, columns=columns, compression=None, keep_local=True) as out:
        for dati in tqdm.tqdm(df.itertuples()):
            seq, st, ed = dati[1], dati[2], dati[3]
            if mrnatok.validate_mrna(seq, st, ed) != 0:
                continue

            cret = mrnatok.get_ids(seq, st, ed)
            sample = {
                'ids': cret,
            }
            
            out.write(sample)
            
    print('done tokenizing')
    del df
    free_mem()
    
if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(myrun)