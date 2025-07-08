
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from dataset import generate_sequence
from ntm_model import NTM

random.seed(42)
torch.manual_seed(42)

input_size = 8
output_size = 8
controller_size = 150
seq_len = 6
epochs = 2000
lr = 5e-4
N = 128
M = 40

ntm = NTM(input_size, output_size, controller_size, N=N, M=M)
optimizer = optim.Adam(ntm.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
loss_fn = nn.BCELoss()
loss_history = []
acc_history = []

for epoch in range(1, epochs + 1):
    input_seq, target_seq = generate_sequence(seq_len, input_size)

    ntm.reset()
    state = (
        torch.zeros(controller_size),
        torch.zeros(controller_size)
    )

    for t in range(seq_len):
        _, state = ntm(input_seq[t], state)

    outputs = []
    blank = torch.zeros(input_size)
    for t in range(seq_len):
        out, state = ntm(blank, state)
        state = (state[0].detach(), state[1].detach())
        outputs.append(out)

    outputs = torch.stack(outputs)
    loss = loss_fn(outputs, target_seq)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == target_seq).float().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())
    acc_history.append(accuracy.item())

    if epoch % 500 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, Accuracy: {accuracy.item()*100:.2f}%")

torch.save(ntm.state_dict(), "ntm_trained.pth")

# Gráfico de Loss
plt.figure(figsize=(10, 4))
plt.plot(loss_history, color='blue')
plt.title("Loss durante o Treinamento")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico de Acurácia
plt.figure(figsize=(10, 4))
plt.plot(acc_history, color='green')
plt.title("Acurácia durante o Treinamento")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar exemplo após treino
test_seq, target_seq = generate_sequence(seq_len, input_size)
ntm.reset()
state = (torch.zeros(controller_size), torch.zeros(controller_size))

for t in range(seq_len):
    _, state = ntm(test_seq[t], state)

blank = torch.zeros(input_size)
predicted = []
for t in range(seq_len):
    out, state = ntm(blank, state)
    predicted.append((out > 0.5).int())

predicted = torch.stack(predicted)

print("\n================ EXEMPLO DE RESULTADO =================")
print("Entrada:")
print(test_seq.int())
print("Saída esperada (bits invertidos):")
print(target_seq.int())
print("Saída prevista pela NTM:")
print(predicted)
print("=======================================================")
