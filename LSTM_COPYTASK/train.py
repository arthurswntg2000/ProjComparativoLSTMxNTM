
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from dataset import generate_copy_task
from lstm_model import SimpleLSTM

def train():
    random.seed(42)
    torch.manual_seed(42)

    input_size = 8
    hidden_size = 64
    output_size = 8
    seq_len = 5
    total_len = seq_len * 2 + 1
    epochs = 2000
    lr = 1e-3
    batch_size = 4

    model = SimpleLSTM(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    acc_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0

        for _ in range(batch_size):
            x, y = generate_copy_task(seq_len, input_size)
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            output = model(x)
            loss = criterion(output, y)
            pred = (torch.sigmoid(output) > 0.5).float()
            acc = (pred == y).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        loss_history.append(total_loss / batch_size)
        acc_history.append(total_acc / batch_size)

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / batch_size:.6f} | Accuracy: {total_acc / batch_size:.2%}")

    torch.save(model.state_dict(), "lstm_copy_task.pth")

    # Gráfico de Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='red')
    plt.title("Loss por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico de Acurácia
    plt.figure(figsize=(8, 4))
    plt.plot(acc_history, color='blue')
    plt.title("Acurácia por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model.eval()
    with torch.no_grad():
        x, y = generate_copy_task(seq_len, input_size)
        output = model(x.unsqueeze(0))
        pred = (torch.sigmoid(output) > 0.5).int().squeeze(0)

        print("="*40)
        print("\nEntrada:")
        print(x.int())
        print("Saída esperada:")
        print(y.int())
        print("Saída prevista:")
        print(pred)


if __name__ == "__main__":
    train()
