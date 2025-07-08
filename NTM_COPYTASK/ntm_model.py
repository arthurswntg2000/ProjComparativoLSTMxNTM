
import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, input_size, output_size, controller_size, memory_size):
        super().__init__()
        self.lstm = nn.LSTMCell(input_size + memory_size, controller_size)
        self.fc = nn.Linear(controller_size, output_size)

    def forward(self, x, prev_state):
        h, c = self.lstm(x, prev_state)
        out = torch.tanh(self.fc(h))
        return out, (h, c)

class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size, N=128, M=40):
        super().__init__()
        self.N, self.M = N, M
        self.controller = Controller(input_size, output_size, controller_size, memory_size=M)
        self.register_buffer('memory', torch.randn(N, M) * 0.01)
        self.read_vector = torch.zeros(M)
        self.read_weights = F.softmax(torch.zeros(N), dim=0)

        self.erase = nn.Linear(controller_size, M)
        self.add = nn.Linear(controller_size, M)
        self.key = nn.Linear(controller_size, M)
        self.beta = nn.Linear(controller_size, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset(self):
        self.memory = torch.randn_like(self.memory) * 0.01
        self.read_vector = torch.zeros_like(self.read_vector)

    def cosine_similarity(self, key):
        k = key / (key.norm() + 1e-8)
        mem_norm = self.memory / (self.memory.norm(dim=1, keepdim=True) + 1e-8)
        return torch.matmul(mem_norm, k)

    def address_memory(self, h):
        key = torch.tanh(self.key(h))
        beta = F.softplus(self.beta(h)) + 1e-8
        sim = self.cosine_similarity(key)
        return F.softmax(beta * sim, dim=0)

    def read(self, w):
        return torch.matmul(w.unsqueeze(0), self.memory).squeeze(0)

    def write(self, w, h):
        erase = torch.sigmoid(self.erase(h))
        add = torch.tanh(self.add(h))
        memory = self.memory.clone()
        memory = memory * (1 - w.unsqueeze(1) * erase.unsqueeze(0)) + w.unsqueeze(1) * add.unsqueeze(0)
        self.memory = memory

    def forward(self, x, prev_state):
        x = torch.cat([x, self.read_vector], dim=-1)
        out, state = self.controller(x, prev_state)
        h = state[0]
        w = self.address_memory(h)
        self.write(w, h)
        self.read_vector = self.read(w)
        return out, state
