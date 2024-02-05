# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
#%%
import os
os.environ['DATA_OPTIMIZER_DATA_CACHE_FOLDER'] = './cache/'

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

import pandas as pd
import gc
import glob
import os
import shutil
import zstandard as zstd
from lightning.data.streaming import DataChunkRecipe, DataProcessor
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from mrna_utils import *



#%%
import pandas as pd

#%%
import os
class MRNADataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: MRNATOK, chunk_size: int, fn_in):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.fn_in = fn_in

    def prepare_structure(self, input_dir):
        print(f'loading {self.fn_in}')
        df = pd.read_parquet(self.fn_in)
        chunk_size_rows = 10000
        ret = []
        os.makedirs(input_dir, exist_ok=True)
        for i in range(0, len(df), chunk_size_rows):
            fn_out = input_dir + f'/{i}.parquet.snappy'
            df_chunk = df.iloc[i:(i+chunk_size_rows), :]

            # Process the chunk here
            if len(df_chunk)>0:
                df_chunk.to_parquet(fn_out)
                ret.extend([fn_out])
            del df_chunk
        del df
        free_mem()
        print(len(ret))
        print(ret[0])
        return ret
    
    def prepare_item(self, fn):
        subdf = pd.read_parquet(fn)
        for dati in subdf.itertuples():
            seq, st, ed = dati[1], dati[2], dati[3]
            if self.tokenizer.validate_mrna(seq, st, ed) != 0:
                continue

            cret = self.tokenizer.get_ids(seq, st, ed)
            yield cret
        del subdf
        free_mem()
def prepare(
    input_dir_root: str = "/data2/data/raw_seqs_cds_pos/",
    output_dir_root: str = "/data2/data/raw_seqs_cds_pos/tensor_after_tok/",
    mrna_cate: str = "rodents",
    chunk_size: int = (2048 * 16384),
    fast_dev_run: bool = False,
) -> None:
    tokenizer = MRNATOK()
    input_file = Path(input_dir_root + f'{mrna_cate}.parquet.snappy')
    input_dir = input_dir_root + '/tmp/' + f'{mrna_cate}/'
    output_dir = Path(output_dir_root + f'{mrna_cate}/')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    data_recipe = MRNADataRecipe(tokenizer=tokenizer, chunk_size=chunk_size, fn_in = input_file)
    data_processor = DataProcessor(
        input_dir=input_dir,
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=8,
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    shutil.rmtree(input_dir) # remove temp dir 

if __name__ == "__main__":

    from jsonargparse import CLI
    CLI(prepare)
