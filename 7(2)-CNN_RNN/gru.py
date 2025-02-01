import torch
from torch import nn, Tensor


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
        '''
        Reset 게이트:
            r_t = sigmoid(Wr @ h + Ur @ x): sigmoid(h @ Wr.T + x @ Ur.T)
        Update 게이트:
            u_t = sigmoid(Wu @ h + Uu @ x): sigmoid(h @ Wu.T + x @ Uu.T)
        후보 은닉 상태:
            h_t(tilde) = tanh((h * r_t) @ W.T + x @ U.T)
        최종 은닉 상태:
        h_t = (1 - u_t) * h + u_t * h_t(tilde)
        '''
        r_t = torch.sigmoid(self.linear_Wr(h) + self.linear_Ur(x))
        u_t = torch.sigmoid(self.linear_Wu(h) + self.linear_Uu(x))
        h_t_tilde = torch.tanh(self.linear_W(h * r_t) + self.linear_U(x))
        h_t = (1 - u_t) * h + u_t * h_t_tilde
        
        return h_t


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        '''GRUcell을 여러 개 반복하여 전체 시퀀스 처리'''
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_length, d_model = inputs.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_length): # 시퀀스 길이만큼 반복하여 GRUcell 실행
            h_t = self.cell(inputs[:,t,:], h_t) # 현재 시간 t의 입력인자와 이전 은닉 상태 h_t
        
        return h_t # 최종 은닉 상태를 return