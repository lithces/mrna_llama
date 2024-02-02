#%%
import os

root = '/data2/data/raw_seqs_cds_pos/np_after_tok_mds/'
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

for ci in lst_cate:
    # os.system(f'cd {root}/{ci};unzstd -f *.zstd;')
    os.system(f'cd {root}/{ci};rm -rf *.zstd;')

# %%
