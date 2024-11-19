from dataParser import read_tokens, read_tokens_idx, build_vocab
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from tqdm import tqdm
from model import GPT
import os, argparse
from dataParser import tokens_dataloader
from checkpointManager import CheckpointManager




def main(config):
    torch.manual_seed(config.seed)

    # 准备数据集
    print('Loading tokens')
    tokens = read_tokens(config.data_path)
    print('Loading Vocab')
    vocab = build_vocab(tokens)
    
    assert config.vocab_size >= len(vocab), f'vocab_size({config.vocab_size}) < len(vocab)({len(vocab)})'
    print(f"Data Loaded, vocab size = {config.vocab_size}")

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        print('cuda not available, use cpu instead')
    device = torch.device(config.device)
    pad_idx = vocab['<pad>'] 
    sep_idx = vocab['<sep>']

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
    model.eval()
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        model, _, _ = checkpoint_manager.load(config.resume_from, model)
    
    print('Start Chatting')
    # while True:
    #     prompt = input('>>>')
    #     if prompt == 'exit':
    #         break
    #     prompt_token = list(prompt) + ['<sep>']
    #     # print('prompt_token', prompt_token)
    #     idx = read_tokens_idx([prompt_token], vocab, show_bar=False)
    #     # print(idx)
    #     idx = torch.tensor(idx, dtype=torch.int64).to(device)
    #     y = model.generate(idx, max_new_tokens=100, stop_token=pad_idx, temperature=1.0)
    #     for c in y[0][len(prompt_token)+1:]:
    #         c = vocab.to_tokens(c)
    #         if c != '<sep>':
    #             print(f'{c}', end='')
    #         else:
    #             print()
    #     print()
    history = None
    # initial_prompt = ['在吗', '在滴，又来找我聊天啦', '小姐姐真好看哟', '小哥哥帅捏']
    # for p in initial_prompt:
    #     p = list(p) + ['<sep>']
    #     p = read_tokens_idx([p], vocab, seq_len=len(p), show_bar=False)
    #     p = torch.tensor(p, dtype=torch.int64).to(device)
    #     if history is None:
    #         history = p
    #     else:
    #         history = torch.cat((history, p), dim=-1)

    while True:
        # history = None # 清空历史记录
        prompt = input('>>>')
        if prompt == 'exit':
            break
        prompt_token = list(prompt) + ['<sep>']
        if len(prompt_token) > config.block_size:
            prompt_token = prompt_token[-config.block_size:]
        prompt_idx = read_tokens_idx([prompt_token], vocab, seq_len=len(prompt_token), show_bar=False)
        prompt_idx = torch.tensor(prompt_idx, dtype=torch.int64).to(device)
        if history is None:
            history = prompt_idx
        else:
            history = torch.cat((history, prompt_idx), dim=-1)
        for i in range(100):
            if history.shape[1] > config.block_size:
                history = history[:, -config.block_size:]
            pred = model.generate_once(history, temperature=1.0)
            c = vocab.to_tokens(pred)
            
            if c == '<pad>' or c == '<sep>':
                history = torch.cat([history, torch.tensor([[sep_idx]]).to(device)], dim=1)
                break
            else:
                print(f'{c}', end='')
            # else:
            #     print()
            history = torch.cat([history, pred], dim=1)
        print()
        # print('debug------')
        # for i in history[0]:
        #     print(vocab.to_tokens(i), end='')
        # print()
    
        
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
    
    
