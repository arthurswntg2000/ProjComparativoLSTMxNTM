# LSTM - Tarefa de Inversão de Bits

Este projeto utiliza uma rede LSTM simples para aprender a inverter sequências binárias.

## 📌 Objetivo

Treinar um modelo LSTM que receba uma sequência de bits como entrada e produza a sequência invertida como saída (espelhada horizontalmente). Por exemplo:

Entrada: `[1, 0, 1, 1]`  
Saída esperada: `[1, 1, 0, 1]`

## 🧠 Arquitetura

- **`SimpleLSTM`**: Implementação de uma rede com uma camada LSTM seguida de uma camada totalmente conectada (`Linear`).

```python
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        ...
    def forward(self, x):
        ...
