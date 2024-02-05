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
data_out_root = '/data2/data/raw_seqs_cds_pos/np_after_tok/'



lst_fn = glob.glob(f'{data_in}/*.parquet.snappy')

# A dictionary mapping input fields to their data types
columns = {
    'ids': 'pkl',
}

# Shard compression, if any
compression = 'zstd'
#%%
import numpy as np
import numba
from numba.typed import Dict as nbDict

class MRNATOK:
    def __init__(self, lst_nuc=list('XAUGC'), 
                 chars_all = [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]):
        self.lst_nuc = lst_nuc
        self.chars_all = chars_all
        self.map_nuc = self.gen_mapping_for_nuc(offset=0)

        # too slow with dict
        # self.nbdic_nuc = nbDict()
        # for kk, vv in self.map_nuc.items():
        #     self.nbdic_nuc[kk] = vv
        self.array_map_nuc = np.array([0] * 256)
        for kk, vv in self.map_nuc.items():
            self.array_map_nuc[ord(kk)] = vv

        self.len_nuc = len(self.lst_nuc)
        # lst_special = ['first_seg_begin', 'first_seg_end', 'codon_begin', 'codon_end', 'last_seg_begin', 'last_seg_end']
        lst_special = ['<5b>', '<cb>', '<3b>', '<3e>']

        self.map_special = dict(zip(lst_special, np.arange(len(lst_special))))
        self.offset_first = len(lst_special)
        self.offset_codon = self.offset_first + self.len_nuc  
        self.offset_last = self.offset_codon + (self.len_nuc**3)
        self.maxlen = 6000

        # build mapping between tok and id
        lst_tok = []
        lst_tok.extend(lst_special)
        lst_tok.extend(lst_nuc)

        for ei in self.lst_nuc:
            for ej in self.lst_nuc:
                for ek in self.lst_nuc:
                    lst_tok.extend([f'{ei}{ej}{ek}'])
        lst_tok.extend(lst_nuc)
        self.lst_tok = lst_tok
        self.map_id_tok = dict(zip(np.arange(len(lst_tok)), lst_tok))

    def gen_mapping_for_nuc(self, offset = 0):
        ret =  dict(zip(self.lst_nuc, np.arange(len(self.lst_nuc))+offset))
        ret['T'] = ret['U']
        for ci in self.chars_all: # for unknown chars
            if ci not in ret:
                ret[ci] = ret['X'] 
    
        return ret

    @staticmethod
    @numba.jit(nopython=True)
    def triple2id_numba(t, m, Q):
        '''
            m = self.map_nuc
            Q = self.len_nuc
        '''
        return m[ord(t[0])]*Q*Q + m[ord(t[1])]*Q + m[ord(t[2])]
    
    @staticmethod
    def triple2id_py(t, m, Q):
        '''
            m = self.map_nuc
            Q = self.len_nuc
        '''
        return m[t[0]]*Q*Q + m[t[1]]*Q + m[t[2]]

    def triple2id(self, t):
        m = self.map_nuc
        Q = self.len_nuc
        return self.triple2id_py(t, m, Q)
        # too slow
        # m = self.array_map_nuc
        # Q = self.len_nuc
        # return self.triple2id_numba(t, m, Q)


    def get_ids(self, seq, st, ed):
        ret_first = [self.map_nuc[xi]+self.offset_first for xi in seq[:st]]
        ret_last = [self.map_nuc[xi]+self.offset_last for xi in seq[ed:]]
        ret_codon = [self.triple2id(seq[ti:(ti+3)]) + self.offset_codon for ti in range(st, ed, 3)] # TODO
        return np.array([self.map_special['<5b>']] + ret_first \
                        + [self.map_special['<cb>']] + ret_codon  \
                        + [self.map_special['<3b>']] + ret_last \
                        + [self.map_special['<3e>']], dtype=np.uint8)


    def validate_mrna(self, seq, st, ed):
        '''
        return: 0 for mrna
        
        '''
        if ( (ed-st)%3 !=0): # not bad codon length
            return 1
        if ( len(seq)>self.maxlen): # too long
            return 2
        if not (seq[st: (st+3)] in ('AUG', 'ATG')): # bad codon start
            return 3
        if not (seq[(ed-3):ed] in ('UAG','UAA','UGA', 'TAG', 'TAA', 'TGA')): # bad codon ending
            return 4 
        return 0




mrnatok = MRNATOK()


#%%
# test code below
if 1:
    tseq = 'dfAUAUGATCUACUGAfgshfAUC'
    tst, ted = 4, len(tseq)-8
    # mrnatok.validate_mrna(tseq, tst, ted)
    ret = mrnatok.get_ids(tseq, tst, ted)
    decode_ret = [mrnatok.map_id_tok[xi] for xi in ret]
    print(tseq)
    print(decode_ret)

#%%
import tqdm

for fi in lst_fn:
    basename = os.path.basename(fi)
    mytype = basename.split('.')[0]
    print('---- loading', mytype)        

    free_mem()

    df = pd.read_parquet(fi)
    print('done loading, tokenizing')
    data_out_path = f'{data_out_root}/{mytype}/'
    shutil.rmtree(data_out_path)
    with MDSWriter(out=data_out_path, columns=columns, compression=compression, keep_local=True) as out:
        for dati in tqdm.tqdm(df.itertuples()):
            seq, st, ed = dati[1], dati[2], dati[3]
            if mrnatok.validate_mrna(seq, st, ed) != 0:
                continue

            cret = mrnatok.get_ids(seq, st, ed)
            # sample = {
            #     'ids': cret,
            #     'class': mytype,
            # }
            sample = {
                'ids': cret,
            }
            
            out.write(sample)
            
    print('done tokenizing')
    del df
    free_mem()
# %%
if 1:
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
