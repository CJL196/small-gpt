from torch import nn
import torch
from torch.nn import functional as F

# class LayerNorm(nn.Module):
#     """
#     层规范化操作
#     """
#     def __init__(self, sequence_len:int, num_hiddens:int, use_bias=False, **kwargs):
#         super(LayerNorm, self).__init__(**kwargs)
#         self.norm_shape = [sequence_len, num_hiddens]
#         self.ln = nn.LayerNorm(self.norm_shape, use_bias)

#     def forward(self, X):
#         return self.ln(X)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, sequence_len:int, num_hiddens:int, use_bias=False, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_hiddens))
        self.bias = nn.Parameter(torch.zeros(num_hiddens)) if use_bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
if __name__ == '__main__':
    """
    Layernorm 将张量的后几个维度数据标准化为均值为0，方差为1
    Layernorm 初始化指定参数的方法如下：
    """
    a = torch.randn(20, 30, 15)
    ln1 = nn.LayerNorm(15)
    print(ln1(a).shape)
    ln2 = nn.LayerNorm([30, 15])
    print(ln2(a).shape)
    ln2 = LayerNorm(sequence_len=30, num_hiddens=15)
    print(ln2(a).shape)
    ln3 = nn.LayerNorm([20, 30, 15])
    print(ln3(a).shape)
    # 标准化为均值为0，方差为1的操作如下：
    def normalize(a: list)->list:
        avg = sum(a)/len(a)
        sigma = 0
        for i in a:
            sigma += (i-avg)**2
        sigma = (sigma/len(a))**0.5
        b = [(i-avg)/sigma for i in a]
        return b
    # 检验LayerNorm的功能
    print('-----------')
    print('LayerNorm的结果：')
    X = torch.tensor([
                [[1, 1, 4],
                [5, 1, 4]],
                [[1, 9, 1],
                [9, 8, 1]],
                [[0, 1, 1],
                [4, 5, 1]],
                [[4, 1, 9],
                [1, 9, 8]]
                ], dtype=torch.float32)
    print(X.shape)
    ln = nn.LayerNorm(3)
    print(ln(X))
    print('对最后一个维度的一个向量算normalize的结果：')
    print(normalize([1, 1, 4]))
    print('-----------')
    print('LayerNorm的结果：')
    ln = LayerNorm(sequence_len=2, num_hiddens=3)
    print(ln(X))
    print('对最后两个维度算normalize的结果：')
    print(normalize([1, 1, 4, 5, 1, 4]))
    
    