from torch import nn
import torch, math
from torch.nn import functional as F

class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, seq_len, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        """
        mask是一个下三角矩阵
        >>> seq_len = 5
        >>> torch.tril(torch.ones(seq_len, seq_len).view(1, seq_len, seq_len))
        tensor([[[1., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]]])
        """
        self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len).view(1, seq_len, seq_len)))

    
    def forward(self, queries, keys, values):
        # queries的形状：(batch_size，序列长度，num_hiddens)
        # keys的形状：(batch_size，序列长度，num_hiddens)
        # values的形状：(batch_size，序列长度，num_hiddens)
        
        num_hiddens = queries.shape[-1]
        # 需要交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(num_hiddens)
        # scores形状：(batch_size, 序列长度，序列长度)
        scores = self.masked_softmax(scores)
        # print(f'scores.shape={scores.shape}, value.shape={values.shape}')

        if self.training: # Don't always dropout
            return torch.bmm(self.dropout(scores), values)
        else:
            return torch.bmm(scores, values)

    def masked_softmax(self, scores):
        """
        scores形状: (batch_size, 序列长度，序列长度)
        Example:
        >>> scores=torch.randn(3, 4, 4)
        >>> masked_softmax(scores)
        ```
        tensor([[[[1.0000, 0.0000, 0.0000, 0.0000],
                [0.2736, 0.7264, 0.0000, 0.0000],
                [0.1757, 0.7120, 0.1123, 0.0000],
                [0.0867, 0.5268, 0.2397, 0.1468]],

                [[1.0000, 0.0000, 0.0000, 0.0000],
                [0.9231, 0.0769, 0.0000, 0.0000],
                [0.3515, 0.0802, 0.5683, 0.0000],
                [0.1712, 0.3664, 0.1514, 0.3110]],

                [[1.0000, 0.0000, 0.0000, 0.0000],
                [0.1340, 0.8660, 0.0000, 0.0000],
                [0.5086, 0.4214, 0.0699, 0.0000],
                [0.0665, 0.4997, 0.2184, 0.2154]]]])
        ```
        """
        _, T, T = scores.size()
        scores = scores.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
        """
        scores中mask矩阵为0的位置会被填充-inf，求softmax之后变为0
        """
        scores = F.softmax(scores, dim=-1)
        return scores

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, num_hiddens, num_heads, seq_len, dropout, use_bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        assert num_hiddens % num_heads == 0
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout, seq_len)
        self.output_drop = nn.Dropout(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)
        self.out_net = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)

    def forward(self, x):
        # queries，keys，values的形状:
        # (batch_size，序列长度，num_hiddens)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，序列长度，num_hiddens/num_heads)
        queries, keys, values = x, x, x
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，序列长度，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        # output_concat的形状:(batch_size，序列长度，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.output_drop(self.out_net(output_concat))


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])



def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_hiddens, num_heads = 100, 5
    batch_size, seq_len = 16, 50
    attention = MultiHeadAttention(num_hiddens, num_heads, seq_len, 0.5).to(device)
    attention.eval()
    
    X = torch.ones((batch_size, seq_len, num_hiddens)).to(device)
    print(X.shape)
    print(attention(X).shape)

"""
torch.Size([16, 50, 100])
torch.Size([16, 50, 100])
"""