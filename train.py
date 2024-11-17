from dataParser import read_tokens, read_tokens_idx, build_vocab
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from model import GPT
#
path = 'data/train.txt'
batch_size = 8
block_size=1024
vocab_size=7587
n_layer=12
n_head=12
n_embd=768
dropout=0.0
seed = 114514
bias=True
tensorboard_path = './tensorboard'
#

global_step=0


writer = SummaryWriter(tensorboard_path)

def tokens_dataloader(tokens_idx, batch_size, pad_idx, is_train=True):
    """构造一个PyTorch数据迭代器"""
    from torch.utils import data
    add = torch.full((tokens_idx.shape[0], 1), pad_idx, dtype=torch.int64)
    y = torch.cat([tokens_idx[:, 1:], add], dim=-1)
    dataset = data.TensorDataset(tokens_idx, y)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def main():
    global vocab_size, global_step
    # 准备数据集
    print('Loading tokens')
    tokens = read_tokens(path)
    print('Loading Vocab')
    vocab = build_vocab(tokens)
    print('Loading tokens_idx tensor, which may take some time')
    tokens_idx = torch.tensor(read_tokens_idx(tokens, vocab, seq_len=block_size), dtype=torch.int64)
    vocab_size = len(vocab)
    print(f"Data Loaded, vocab size = {vocab_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_idx = vocab['<pad>'] 

    data_loader = tokens_dataloader(tokens_idx, batch_size, pad_idx, is_train=True)
    
    # 加载模型
    model = GPT(
        block_size,
        vocab_size,
        n_layer,
        n_head,
        n_embd,
        dropout,
        bias, 
    ).to(device)
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        print(f'logits.shape={logits.shape}, loss={loss.item()}')
        writer.add_scalar('loss', loss.item(), global_step)
        loss.backward()
        global_step += 1
        

        
if __name__ == '__main__':
    main()
    
    
