
import torch
from ntm_model import NTM
from dataset import generate_sequence

input_size = 8
output_size = 8
controller_size = 100
seq_len = 10

ntm = NTM(input_size, output_size, controller_size)
ntm.load_state_dict(torch.load("ntm_trained.pth", map_location=torch.device('cpu')))
ntm.eval()

with torch.no_grad():
    test_seq, target_seq = generate_sequence(seq_len, input_size)
    ntm.reset()
    state = (
        torch.zeros(controller_size),
        torch.zeros(controller_size)
    )

    for t in range(seq_len):
        _, state = ntm(test_seq[t], state)

    blank = torch.zeros(input_size)
    predicted = []
    for t in range(seq_len):
        out, state = ntm(blank, state)
        predicted.append((out > 0.5).int())

    predicted = torch.stack(predicted)
    print("Entrada:")
    print(test_seq.int())
    print("Saída esperada (bits invertidos):")
    print(target_seq.int())
    print("Saída prevista:")
    print(predicted)
