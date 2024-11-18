from dataParser import read_tokens, read_tokens_idx, build_vocab, tokens_dataloader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm import tqdm
from model import GPT
import os, argparse
from checkpointManager import CheckpointManager

global_step = 0
global_epoch = 0

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

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        print('cuda not available, use cpu instead')
    device = torch.device(config.device)
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
    optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        model, global_step, global_epoch = checkpoint_manager.load(config.resume_from, model, optimizer)
    
    writer = SummaryWriter(config.tensorboard_path)
    
    print('Start Training')
    while global_epoch < config.max_epoch:
        prog_bar = tqdm(data_loader)
        for x, y in prog_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            prog_bar.set_description(f'{global_step}|{global_epoch}|loss={loss.item():.8}')
            writer.add_scalar('loss', loss.item(), global_step)
            loss.backward()
            global_step += 1
            optimizer.step()
            
            if global_step % config.save_freq == 0:
                checkpoint_manager.save(model, config, global_step, global_epoch, optimizer)
            
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
    
    
