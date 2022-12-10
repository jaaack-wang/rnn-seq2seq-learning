'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Code for creating dataloader in PyTorch
'''
import torch
try:
    from utils import read_data
except:
    pass
from string import ascii_lowercase
from torch.utils.data import Dataset, DataLoader


class Transfrom(Dataset):
    def __init__(self, in_vocab, out_vocab, data):
        self.data = data
        self.iv2idx, self.idx2iv = self._make_map(in_vocab)
        self.ov2idx, self.idx2ov = self._make_map(out_vocab)
    
    def _make_map(self, vocab):
        v2idx = {"<s>": 0, "</s>": 1}
        v2idx.update({v: idx+2 for idx, v in enumerate(vocab)})
        idx2v = {idx: v for v, idx in v2idx.items()}
        return v2idx, idx2v

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        in_seq, out_seq = self.data[index]
        x = [self.iv2idx.get(
            i_s, len(self.iv2idx)) for i_s in in_seq]
        y = [self.ov2idx.get(
            o_s, len(self.ov2idx)) for o_s in out_seq]
        
        return [0] + x + [1], [0] + y + [1]


def collate_fn(batch, max_seq_len=None):
    N = len(batch)
    X, Y = zip(*batch)
    in_max_len = max([len(x) for x in X])
    out_max_len = max([len(y) for y in Y])
    
    if max_seq_len and max_seq_len > in_max_len:
        inputs = torch.ones(max_seq_len, N).long()
    else:
        inputs = torch.ones(in_max_len, N).long()
    
    if max_seq_len and max_seq_len > out_max_len:
        outputs = torch.ones(max_seq_len, N).long()
    else:
        outputs = torch.ones(out_max_len, N).long()

    for idx, (x, y) in enumerate(batch):
        inputs[:len(x), idx] = torch.Tensor(x).long()
        outputs[:len(y), idx] = torch.Tensor(y).long()

    return inputs, outputs


def create_dataloader(data,
                      in_vocab=ascii_lowercase,
                      out_vocab=ascii_lowercase,
                      batch_size=256,
                      shuffle=False,
                      collate_fn=collate_fn,
                      max_seq_len=None):
    
    if isinstance(data, str):
        data = read_data(data)
        
    if max_seq_len:
        c_f = lambda batch: collate_fn(batch, max_seq_len)
    else:
        c_f = collate_fn
    
    dataset = Transfrom(in_vocab, out_vocab, data)
    dataloader = DataLoader(dataset, batch_size, 
                            shuffle, collate_fn=c_f)
    return dataloader
