#%%
import numpy as np
from streaming import MDSWriter

import pandas as pd
import gc
import glob
import os
import shutil
import numba
import torch
import re
mydtype=np.uint8
#%% first, extract stats to distributed parquet files, free memory
import ctypes
def free_mem():
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)



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
        lst_codon = []
        for ei in self.lst_nuc:
            for ej in self.lst_nuc:
                for ek in self.lst_nuc:
                    lst_codon.extend([f'{ei}{ej}{ek}'])

        lst_tok.extend(lst_codon)        
        lst_tok.extend(lst_nuc)
        self.lst_tok = lst_tok
        self.map_id_tok = dict(zip(np.arange(len(lst_tok)), lst_tok))
        # self.map_tok_id = dict(zip(lst_tok, np.arange(len(lst_tok))))
        self.map_codon_id = dict(zip(lst_codon, np.arange(len(lst_codon)))) # starting from zero

        self.pattern = re.compile(f"[^{re.escape(str(lst_nuc))}]")

    def clean_str(self, str_in):
        # Use re.sub() to replace the matched characters with the replacement character
        output_string = re.sub("T", 'U', str_in)
        output_string = re.sub(self.pattern, 'X', output_string)
        return output_string

    def gen_mapping_for_nuc(self, offset = 0):
        ret =  dict(zip(self.lst_nuc, np.arange(len(self.lst_nuc))+offset))
        ret['T'] = ret['U']
        for ci in self.chars_all: # for unknown chars
            if ci not in ret:
                ret[ci] = ret['X'] 
    
        return ret

    @staticmethod
    def triple2id_numba(t, m, Q):
        '''
            m = self.map_nuc
            Q = self.len_nuc
        '''
        return m[ord(t[0])]*Q*Q + m[ord(t[1])]*Q + m[ord(t[2])]
    
    @staticmethod
    def triple2id_py(t, m):
        '''
        # please ensure that t is cleansed
        m = self.map_codon
        '''
        # return m[t[0]]*Q*Q + m[t[1]]*Q + m[t[2]]
        return m[t]

    def triple2id(self, t):
        '''
        '''
        m = self.map_codon_id
        Q = self.len_nuc
        return self.triple2id_py(t, m, Q)
        # too slow
        # m = self.array_map_nuc
        # Q = self.len_nuc
        # return self.triple2id_numba(t, m, Q)

    def get_ids_py(self, seq, st, ed):
        seq = self.clean_str(seq)
        ret_first = [self.map_nuc[xi]+self.offset_first for xi in seq[:st]]
        ret_last = [self.map_nuc[xi]+self.offset_last for xi in seq[ed:]]
        ret_codon = [self.triple2id_py(seq[ti:(ti+3)], self.map_codon_id) + self.offset_codon for ti in range(st, ed, 3)] # TODO
        return np.array([self.map_special['<5b>']] + ret_first \
                        + [self.map_special['<cb>']] + ret_codon  \
                        + [self.map_special['<3b>']] + ret_last \
                        + [self.map_special['<3e>']], dtype=np.uint8)

    def get_ids_pythran(self, seq, st, ed):
        r = mrna_tok_pythran.get_ids(seq, st, ed, \
                self.map_nuc, self.len_nuc, self.map_special, self.offset_first, self.offset_codon, self.offset_last)
        return np.array(r, dtype=np.uint8)

    def get_ids(self, seq, st, ed):
        # return self.get_ids_pythran(seq, st, ed)
        return self.get_ids_py(seq, st, ed)

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
