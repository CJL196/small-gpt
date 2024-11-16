from .block import DecoderBlock
from .layernorm import LayerNorm
import torch, math
from torch import nn
from torch.nn import functional as F

class GPT(nn.Module):
    def __init__(self, block_size:int, vocab_size:int, n_layer:int, n_head:int, n_embd:int, dropout:float, bias:bool):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd), # token embedding
            wpe = nn.Embedding(block_size, n_embd), # position embedding
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([DecoderBlock(block_size, n_embd, n_head, bias, dropout) for _ in range(n_layer)]),
            ln_f = LayerNorm(block_size, n_embd, bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        # weight tying, 参考https://blog.csdn.net/sinat_39448069/article/details/121744119
        self.transformer.wte.weight = self.lm_head.weight
        # 初始化权重
        self.apply(self._init_weights)
        # 针对 GPT-2 模型中 残差投影层（Residual Projections） 的特殊初始化方法, 提高模型的训练稳定性, 参考 GPT-2 论文
        for pn, p in self.named_parameters():
            if pn.endswith('out_net.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        device = x.device
        batch_size, seq_len = x.shape
        assert seq_len <= self.block_size, f"Cannot forward sequence of length {seq_len}, block size is only {self.block_size}"
        # 获取位置编码和token编码
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(x) # tok_emb.shape=(batch_size, seq_len, n_embd)
        pos_emb = self.transformer.wpe(pos) # pos_emb.shape=(seq_len, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        # decoder block 推理
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None: # 训练
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: # 推理
            logits = self.lm_head(x[:, [-1], :]) # 取最后一个token的logits
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        """
        x.shape = (batch_size, 上下文长度)
        预测max_new_tokens次，每次都将预测的结果和x拼接
        """
        
        for _ in range(max_new_tokens):
            # 如果上下文太长，需要裁剪至长度为seq_len
            cropped = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
            # 前向推理
            logits, _ = self(cropped)
            # logits.shape = (batch_size, seq_len, vocab_size)
            # 在最后一步提取 logits 并按所需温度进行缩放
            # 温度越高，概率分布更加平滑，低概率的 token 也有较大的机会被选中。文本生成更具创造性，结果更具多样性。
            logits = logits[:, -1, :] / temperature
            # 只考虑前topk
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            # 采样
            pred = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, pred), dim=1)

        return x
    

"""
Example:
```python
import torch
from model import GPT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT(
        block_size=1024,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    ).to(device)
x = torch.randint(0, 50257, (1, 1024)).to(device)
y = torch.randint(0, 50257, (1, 1024)).to(device)
logits, loss = model(x, y)
print(logits.shape)
print(loss)
y = model.generate(x[:, :3], max_new_tokens=10, temperature=1.0, top_k=10)
print(y.shape)
print(y)
```

Output:
```
number of parameters: 123.65M
torch.Size([1, 1024, 50257])
tensor(10.9876, device='cuda:0', grad_fn=<NllLossBackward0>)
torch.Size([1, 13])
tensor([[43652, 43017, 45172, 48030, 33227, 16497, 16497, 46806, 27099, 38971,
         29453, 16830, 16830]], device='cuda:0')
```
"""