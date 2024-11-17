from dataParser import read_tokens, read_tokens_idx, build_vocab
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm import tqdm
from model import GPT
import os, argparse


global_step = 0
global_epoch = 0

def tokens_dataloader(tokens_idx, batch_size, pad_idx, is_train=True):
    """构造一个PyTorch数据迭代器"""
    from torch.utils import data
    add = torch.full((tokens_idx.shape[0], 1), pad_idx, dtype=torch.int64)
    y = torch.cat([tokens_idx[:, 1:], add], dim=-1)
    dataset = data.TensorDataset(tokens_idx, y)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class CheckpointManager:
    """
    checkpoint standard:
    {
        'global_step': int,
        'global_epoch': int,
        'block_size':int,
        'vocab_size':int,
        'n_layer':int,
        'n_head':int,
        'n_embd':int,
        'bias':bool,
        'dataset': str,
        'state_dict': dict,
    }
    """
    def __init__(self, save_root):
        self.save_root = save_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    def load(self, path, model):
        print(f'Loading State Dict from {path}')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        global global_step, global_epoch
        global_step = checkpoint['global_step']
        global_epoch = checkpoint['global_epoch']
        print('state dict config:')
        for k, v in checkpoint.items():
            if k == 'state_dict':
                continue
            print(f'{k}={v}')
        print('-'*6)
        return checkpoint

    def save(self, steps, model, config):
        filename = f'cpt{steps}.pth'
        path = os.path.join(self.save_root, filename)
        checkpoint = {
            'global_step': global_step,
            'global_epoch': global_epoch,
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'bias': config.bias,
            'dataset': path,
            'state_dict': model.state_dict(),
        }
        torch.save(checkpoint, path)


def main(config):
    global global_step, global_epoch

    # 准备数据集
    print('Loading tokens')
    tokens = read_tokens(config.data_path)
    print('Loading Vocab')
    vocab = build_vocab(tokens)
    print('Loading tokens_idx tensor, which may take some time')
    tokens_idx = torch.tensor(read_tokens_idx(tokens, vocab, seq_len=config.block_size), dtype=torch.int64)
    assert config.vocab_size >= len(vocab), f'vocab_size({config.vocab_size}) < len(vocab)({len(vocab)})'
    print(f"Data Loaded, vocab size = {config.vocab_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_idx = vocab['<pad>'] 

    data_loader = tokens_dataloader(tokens_idx, config.batch_size, pad_idx, is_train=True)
    
    # 加载模型
    model = GPT(
        config.block_size,
        config.vocab_size,
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.dropout,
        config.bias, 
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        checkpoint_manager.load(config.resume_from, model)
    
    writer = SummaryWriter(config.tensorboard_path)
    
    print('Start Training')
    while global_epoch < config.max_epoch:
        prog_bar = tqdm(data_loader)
        for x, y in prog_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            prog_bar.set_description(f'loss={loss.item()}')
            writer.add_scalar('loss', loss.item(), global_step)
            loss.backward()
            global_step += 1
            optimizer.step()
            
            if global_step % config.save_freq == 0:
                checkpoint_manager.save(global_step, model, config)
            
        global_epoch += 1
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    args = parser.parse_args()
    # 从 YAML 文件加载配置
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)
    return config


if __name__ == '__main__':
    config = parse_args()
    main(config)
    
    
