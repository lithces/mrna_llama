#pythran export triple2id(str, str:int dict, int)
def triple2id(t, m, Q):
    '''
        t: string of length 3
        m = self.map_nuc
        Q = self.len_nuc
    '''
    return m[t[0]]*Q*Q + m[t[1]]*Q + m[t[2]]

#pythran export get_ids(str, int, int, str:int dict, int, str:int dict, int, int, int)
def get_ids(seq, st, ed, map_nuc, len_valid_nuc, map_special, \
        offset_first, offset_codon, offset_last):
    ret_first = [map_nuc[xi]+offset_first for xi in seq[:st]]
    ret_last = [map_nuc[xi]+offset_last for xi in seq[ed:]]
    ret_codon = [triple2id(seq[ti:(ti+3)], map_nuc, len_valid_nuc) + offset_codon for ti in range(st, ed, 3)] # TODO
    return [map_special['<5b>']] + ret_first \
                    + [map_special['<cb>']] + ret_codon  \
                    + [map_special['<3b>']] + ret_last \
                    + [map_special['<3e>']]
