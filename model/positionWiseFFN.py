from torch import nn
import torch
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, use_bias=False, dropout=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens, bias=use_bias)
        self.gelu   = nn.GELU()
        self.out_net = nn.Linear(ffn_num_hiddens, ffn_num_outputs, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, X):
        return self.dropout(self.out_net(self.gelu(self.dense1(X))))
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ffn = PositionWiseFFN(4, 6, 8).to(device=device)
    print(ffn)
    ffn.eval()
    out = ffn(torch.ones((2, 3, 4), device=device))
    print(out.shape)
        