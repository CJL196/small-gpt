# Transformer Decoder Block for GPT
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layernorm import LayerNorm
from .positionWiseFFN import PositionWiseFFN

class DecoderBlock(nn.Module):
    def __init__(self, sequence_len:int, num_hiddens:int, num_head:int, use_bias:bool, dropout, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.ln1 = LayerNorm(num_hiddens, use_bias)
        self.ln2 = LayerNorm(num_hiddens, use_bias)
        self.attention = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_head, seq_len=sequence_len, dropout=dropout, use_bias=use_bias)
        self.ffn = PositionWiseFFN(ffn_num_input=num_hiddens, ffn_num_hiddens=num_hiddens*4, ffn_num_outputs=num_hiddens, use_bias=use_bias, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

if __name__ == "__main__":
    batch_size, sequence_len, num_hiddens, num_head = 16, 512, 256, 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.ones([batch_size, sequence_len, num_hiddens]).to(device)
    print(x.shape)
    block = DecoderBlock(sequence_len, num_hiddens, num_head, use_bias=False, dropout=0.1).to(device)
    print(block(x).shape)
    """
    运行结果：
    torch.Size([16, 512, 256])
    torch.Size([16, 512, 256])
    """