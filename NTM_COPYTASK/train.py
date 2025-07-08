
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from dataset import generate_copy_task
from ntm_model import NTM

def train():
    random.seed(42)
    torch.manual_seed(42)

    input_size = 8
    output_size = 8
    controller_size = 100
    seq_len = 5
    total_len = seq_len * 2 + 1
    epochs = 2000
    lr = 1e-4
    batch_size = 4

    model = NTM(input_size, output_size, controller_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loss_history = []
    acc_history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0

        for _ in range(batch_size):
            x, y = generate_copy_task(seq_len, input_size)
            model.reset()
            state = (torch.zeros(controller_size), torch.zeros(controller_size))
            outputs = []

            for t in range(total_len):
                out, state = model(x[t], state)
                outputs.append(out)

            outputs = torch.stack(outputs)
            loss = criterion(outputs, y)
            pred = (torch.sigmoid(outputs) > 0.5).float()
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

    torch.save(model.state_dict(), "ntm_copy_task.pth")

    # Gráfico de Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='red')
    plt.title("Loss da NTM por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico de Acurácia
    plt.figure(figsize=(8, 4))
    plt.plot(acc_history, color='green')
    plt.title("Acurácia da NTM por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    model.eval()
    with torch.no_grad():
        x, y = generate_copy_task(seq_len, input_size)
        model.reset()
        state = (torch.zeros(controller_size), torch.zeros(controller_size))
        preds = []

        for t in range(total_len):
            out, state = model(x[t], state)
            preds.append((torch.sigmoid(out) > 0.5).int())

        preds = torch.stack(preds)
        print("\nEntrada (com marcador):")
        print(x.int())
        print("\nSaída esperada:")
        print(y.int())
        print("\nSaída prevista:")
        print(preds)

if __name__ == "__main__":
    train()
