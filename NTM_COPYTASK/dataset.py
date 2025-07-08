
import torch

def generate_copy_task(seq_len=5, input_size=8):
    input_seq = torch.randint(0, 2, (seq_len, input_size)).float()
    sep = torch.zeros(1, input_size)
    blank = torch.zeros(seq_len, input_size)
    x = torch.cat([input_seq, sep, blank], dim=0)
    y = torch.cat([torch.zeros(seq_len + 1, input_size), input_seq], dim=0)
    return x, y
