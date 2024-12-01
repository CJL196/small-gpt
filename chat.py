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
    
    if config.compile:
        print('compiling the model... this might take a while')
        model = torch.compile(model)
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        model, _, _ = checkpoint_manager.load(config.resume_from, model)
    
    print('Start Chatting')
    history = None

    while True:
        # history = None # 清空历史记录
        prompt = input('>>>')
        if prompt == 'exit':
            break
        if prompt == 'flush':
            history = None
            continue
        # 在每句话输入后加上<sep>，引导模型生成对上一句话的回复
        prompt_token = list(prompt) + ['<sep>']
        # 裁剪上下文长度不超过block_size
        if len(prompt_token) > config.block_size:
            prompt_token = prompt_token[-config.block_size:]
        # 将token转为标号
        prompt_idx = read_tokens_idx([prompt_token], vocab, seq_len=len(prompt_token), show_bar=False)
        prompt_idx = torch.tensor(prompt_idx, dtype=torch.int64).to(device)
        # 将当前对话与历史对话拼接，让模型能够利用上下文信息
        if history is None:
            history = prompt_idx
        else:
            history = torch.cat((history, prompt_idx), dim=-1)
        # 模型逐token生成，单次生成上限为100token
        for i in range(100):
            # 裁剪上下文长度不超过block_size
            if history.shape[1] > config.block_size:
                history = history[:, -config.block_size:]
            # 生成单个token，设置温度为0.5
            pred = model.generate_once(history, temperature=0.5)
            c = vocab.to_tokens(pred)
            # 当生成<pad>或<sep>时，表示这句话说完了
            if c == '<pad>' or c == '<sep>':
                history = torch.cat([history, torch.tensor([[sep_idx]]).to(device)], dim=1)
                break
            else:
                print(f'{c}', end='')
            # 更新历史记录
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
    
    
