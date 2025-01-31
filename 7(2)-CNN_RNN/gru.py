import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_Wr = nn.Linear(hidden_size, hidden_size)
        self.linear_Ur = nn.Linear(input_size, hidden_size)
        self.linear_Wu = nn.Linear(hidden_size, hidden_size)
        self.linear_Uu = nn.Linear(input_size, hidden_size)
        self.linear_W = nn.Linear(hidden_size, hidden_size)
        self.linear_U = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        r_t = F.sigmoid(self.linear_Wr(h) + self.linear_Ur(x))
        u_t = F.sigmoid(self.linear_Wu(h) + self.linear_Uu(x))
        h_t_tilde = F.tanh(self.linear_W(h * r_t) + self.linear_U(x))
        h_t = (1 - u_t) * h + u_t * h_t_tilde
        
        return h_t


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_length, d_model = inputs.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_length):
            h_t = self.cell(inputs[:,t,:], h_t)
        
        return self.fc(h_t)