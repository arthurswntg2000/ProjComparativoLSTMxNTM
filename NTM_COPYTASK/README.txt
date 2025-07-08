NTM - Copy Task

Este projeto utiliza uma Máquina de Turing Neural (NTM) para resolver a tarefa de copiar uma sequência de bits.

## Estrutura

- `dataset.py`: Geração de dados para a tarefa de cópia.
- `ntm_model.py`: Implementação da arquitetura da NTM.
- `train.py`: Script de treinamento do modelo.
# NTM - Neural Turing Machine para Copy Task

Este projeto implementa uma **Neural Turing Machine (NTM)** em PyTorch para resolver a **Copy Task**, uma tarefa clássica de aprendizado sequencial. A NTM recebe uma sequência binária de entrada, seguida por um marcador, e deve reproduzir a sequência original na saída.

---

## Descrição da Tarefa

A **Copy Task** consiste em apresentar ao modelo uma sequência binária de tamanho fixo, seguida por um marcador (vetor de controle), e o objetivo do modelo é **copiar a sequência inicial** como saída.

**Exemplo:**

```text
Entrada:
[seq_1]
[seq_2]
[seq_3]
[seq_4]
[seq_5]
[marcador]
[... zeros ...]

Saída esperada:
[... zeros ...]
[... zeros ...]
[... zeros ...]
[... zeros ...]
[... zeros ...]
[... zeros ...]
[seq_1]
[seq_2]
[seq_3]
[seq_4]
[seq_5]

## Execução

```bash
python train.py
