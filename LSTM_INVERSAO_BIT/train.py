
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from dataset import generate_sequence
from lstm_model import SimpleLSTM

def train():
    random.seed(42)
    torch.manual_seed(42)

    input_size = 8
    hidden_size = 64
    output_size = 8
    seq_len = 6
    epochs = 300
    lr = 5e-4
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
            input_seq, target_seq = generate_sequence(seq_len, input_size)
            input_seq = input_seq.unsqueeze(0)  # batch dimension
            target_seq = target_seq.unsqueeze(0)

            output = model(input_seq)
            loss = criterion(output, target_seq)
            predicted = (torch.sigmoid(output) > 0.5).float()
            accuracy = (predicted == target_seq).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy.item()

        loss_history.append(total_loss / batch_size)
        acc_history.append(total_acc / batch_size)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / batch_size:.6f}, Accuracy: {total_acc / batch_size * 100:.2f}%")

    torch.save(model.state_dict(), "lstm_trained.pth")

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, color='blue')
    plt.title("Loss durante o Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(acc_history, color='green')
    plt.title("Acurácia durante o Treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Exemplo
    input_seq, target_seq = generate_sequence(seq_len, input_size)
    model.eval()
    with torch.no_grad():
        output = model(input_seq.unsqueeze(0))
        predicted = (torch.sigmoid(output) > 0.5).int()

    print("\n================ EXEMPLO DE RESULTADO =================")
    print("Entrada:")
    print(input_seq.int())
    print("Saída esperada (bits invertidos):")
    print(target_seq.int())
    print("Saída prevista pela LSTM:")
    print(predicted.squeeze(0))
    print("=======================================================")

if __name__ == "__main__":
    train()
