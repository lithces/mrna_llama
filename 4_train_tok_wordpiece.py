#%%
vocab_size=16384
offset = 0x4e00
N_samp = 200000

#%%
import numpy as np
######### DATASET SETUP, please test the loader before proceed
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
torch.manual_seed(2809)

class TokenTensorDataset(LocalDataset):
    def __init__(self, local, offset=0x4e00):
        super().__init__(local=local)
        self.offset = offset

    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        return ''.join([chr(xy+self.offset) for xy in dat.tolist()])

# 11 classes
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
ds_root = '/home/lithtp/sanofi/mrna/mrna_llm/raw_seqs_cds_pos/np_after_tok_mds_th20000/'


local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
lst_ds = [TokenTensorDataset(local=li, offset=offset) for li in local_dirs]


ds = ConcatDataset(lst_ds)
N = len(ds)
BS=1024
dl = DataLoader(ds, batch_size=BS, shuffle=True)

# %%
for dati in dl:
    dati
    break

# %%
# %%
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=20000))
# tokenizer.normalizer = None
tokenizer.pre_tokenizer = pre_tokenizers.Split(chr(0), "removed")
trainer = trainers.WordPieceTrainer(
    vocab_size=vocab_size, special_tokens=['[UNK]']
)

# %%
import tqdm
def get_lst_strings():
    N_cyc = N_samp // BS
    for ii, di in tqdm.tqdm(enumerate(dl)):
        if ii > N_cyc:
            break
        yield di
lst = get_lst_strings()
#%%
# dat = torch.concat(lst)
tokenizer.train_from_iterator(lst, trainer=trainer)

# %%
tokenizer.save(f"tok_wordpiece_{vocab_size}.json")
#%%
lst = get_lst_strings()
for ss in lst:
    ret = tokenizer.encode(ss[0])
    break
# %%
decoded_string = tokenizer.decode(ret.ids)
a = decoded_string.replace('##', '').replace(' ', '')
# %%
assert(a==ss[0])
# %%
len(decoded_string.split())
# %%
