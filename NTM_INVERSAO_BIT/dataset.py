
import torch

def generate_sequence(seq_len=10, input_size=8):
    input_seq = torch.randint(0, 2, (seq_len, input_size)).float()
    target_seq = 1.0 - input_seq  # Inversão de bits
    return input_seq, target_seq
