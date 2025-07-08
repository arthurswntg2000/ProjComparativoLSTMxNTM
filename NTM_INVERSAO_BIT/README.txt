## NTM - Neural Turing Machine para Tarefa de Inversão de Bits

Este projeto implementa uma **Máquina de Turing Neural (NTM)** usando PyTorch para resolver a tarefa de **inversão de bits em uma sequência binária**. O modelo recebe uma sequência de vetores binários como entrada e deve produzir a saída com os bits invertidos. Ex: `[1, 0, 1] → [0, 1, 0]`.

---

## Estrutura do Projeto

- `ntm_model.py`: Arquivo com a implementação da arquitetura da NTM.
- `dataset.py`: Função `generate_sequence()` para gerar pares de entrada/saída com bits invertidos.
- `main.py` (ou este script): Código de treinamento e avaliação do modelo.
- `ntm_trained.pth`: Arquivo gerado com os pesos treinados da NTM.

---

## Especificações do Modelo

- **Input size**: 8
- **Output size**: 8
- **Tamanho do controlador**: 150
- **Memória N x M**: 128 x 40
- **Função de perda**: `BCELoss` (Binary Cross Entropy)
- **Otimizador**: Adam (learning rate = 5e-4)
- **Scheduler**: Redução do LR a cada 500 épocas

---

## Tarefa

O objetivo do modelo é aprender a **inverter os bits** de uma sequência binária.  
A sequência de entrada é passada para a NTM, que a processa durante a fase de codificação. Em seguida, uma sequência em branco (vetores zero) é fornecida como entrada, e a NTM deve gerar a sequência invertida correspondente.

---

## Resultados

Durante o treinamento, são gerados dois gráficos:
- **Loss durante as épocas**
- **Acurácia durante as épocas**

Ao final, o script imprime um exemplo de entrada, saída esperada e saída prevista pela NTM.

---

## Como Executar

1. Certifique-se de ter o PyTorch instalado:
   ```bash
   pip install torch matplotlib
