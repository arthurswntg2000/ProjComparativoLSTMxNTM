Este projeto tem como objetivo comparar o desempenho da Neural Turing Machine (NTM) com a Long Short-Term Memory (LSTM) em duas tarefas clássicas de memória sequencial: inversão de bits e CopyTask11. As análises envolvem treinamento supervisionado, avaliação quantitativa de perda e acurácia, bem como generalização fora do intervalo visto em treino.

- Modelos Envolvidos

	Neural Turing Machine (NTM)

Modelo neural com memória externa diferenciável.

Capaz de aprender algoritmos e manipular dados sequencialmente.

Arquitetura usada: NTM com controlador LSTM.

	Long Short-Term Memory (LSTM)

Variante das RNNs com memória interna (celular).

Utilizada como baseline comparativo em tarefas com dependência de longo prazo.

Arquitetura padrão com uma ou mais camadas.



	Tarefas

1. Inversão de Bits

Entrada: sequência binária (ex: 10100110)

Saída esperada: sequência invertida (ex: 01100101)

Objetivo: avaliar a habilidade dos modelos em manipular sequências com operações de memória explícita.

2. CopyTask

Entrada: sequência de bits com delimitador (ex: 101010...<delim>)

Saída esperada: repetição exata da sequência anterior.

Avalia retenção e cópia fiel da memória.



	Metodologia


Dataset sintético: gerado aleatoriamente com tamanhos variáveis.

Treinamento supervisionado com:

MSELoss ou CrossEntropyLoss (dependendo da tarefa)

Otimizador: Adam

Avaliação:

Curvas de loss e acurácia

Testes de generalização com sequências maiores

Comparações visuais e numéricas



	Resultados Esperados


Métrica	Tarefa	NTM (%)	LSTM (%)
Acurácia	Inversão de Bits	> 85%	~60–75%
Acurácia	CopyTask11	> 90%	~70–80%
Generalização	Ambas	Boa	Limitada
Convergência	Ambas	Mais lenta	Mais rápida


	
- NTMs tendem a aprender mais lentamente, mas apresentam melhor capacidade de generalização e manipulação explícita de memória.

	

	Visualizações

Curvas de Loss por época

Acurácia por época

Exemplos de entradas e saídas reais dos modelos


	Estrutura de Arquivos

comparativo_ntm_lstm/
├── datasets/
│   ├── inverter_dataset.py
│   └── copytask_dataset.py
├── models/
│   ├── ntm.py
│   └── lstm.py
├── train.py
├── evaluate.py
├── test.py
├── plots/
│   ├── loss_plot.png
│   └── accuracy_plot.png
└── README.md


	Requisitos

Python 3.8+

PyTorch

matplotlib

numpy

	Instalar dependências:

bash
Copiar
Editar
pip install -r requirements.txt

	Como Executar

Treinamento
bash
Copiar
Editar
python train.py --model ntm --task inverter
python train.py --model lstm --task copy
Avaliação
bash
Copiar
Editar
python evaluate.py --model ntm --task inverter
Visualização
Gráficos de loss e acurácia serão salvos na pasta plots/.

- Referências
Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv:1410.5401

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation

- Autor
Arthur Sampaio Pereira
Disciplina: Linguagens Formais e Autômatos – 2025.1
Curso: Engenharia da Computação