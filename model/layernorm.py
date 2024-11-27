from torch import nn
import torch
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, num_hiddens:int, use_bias=False, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_hiddens))
        self.bias = nn.Parameter(torch.zeros(num_hiddens)) if use_bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
if __name__ == '__main__':
    """
    Layernorm 将张量的最后一个维度数据标准化为均值为0，方差为1
    Layernorm 初始化指定参数的方法如下：
    """
    a = torch.randn(20, 30, 15)
    ln2 = LayerNorm(num_hiddens=15)
    print(ln2(a).shape)
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
    print('-----------')
    print('LayerNorm的结果：')
    ln = LayerNorm(num_hiddens=3)
    print(ln(X))
    print('对最后一个维度算normalize的结果：')
    print(normalize([1, 1, 4]))
    print(normalize([5, 1, 4]))
    print(normalize([1, 9, 1]))
    print(normalize([9, 8, 1]))
    
"""
torch.Size([20, 30, 15])
-----------
LayerNorm的结果：
tensor([[[-0.7071, -0.7071,  1.4142],
         [ 0.9806, -1.3728,  0.3922]],

        [[-0.7071,  1.4142, -0.7071],
         [ 0.8429,  0.5620, -1.4049]],

        [[-1.4142,  0.7071,  0.7071],
         [ 0.3922,  0.9806, -1.3728]],

        [[-0.2020, -1.1112,  1.3132],
         [-1.4049,  0.8429,  0.5620]]], grad_fn=<NativeLayerNormBackward0>)
对最后一个维度算normalize的结果：
[-0.7071067811865475, -0.7071067811865475, 1.414213562373095]
[0.98058067569092, -1.3728129459672882, 0.392232270276368]
[-0.7071067811865475, 1.4142135623730951, -0.7071067811865475]
[0.8429272304235246, 0.5619514869490164, -1.404878717372541]
"""