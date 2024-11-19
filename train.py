from dataParser import read_tokens, read_tokens_idx, build_vocab, tokens_dataloader, split_train_test
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch, math
from tqdm import tqdm
from model import GPT
import os, argparse
from checkpointManager import CheckpointManager
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext

global_step = 0
global_epoch = 0

def calc_lr(config, step):
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    elif step < config.decay_steps:
        decay_ratio = (step - config.warmup_steps) / (config.decay_steps - config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.lr - config.min_lr)
    else:
        return config.min_lr

def main(config):
    global global_step, global_epoch

    torch.manual_seed(config.seed)
    # 准备数据集
    print('Loading tokens')
    tokens = read_tokens(config.data_path)
    # 划分训练和测试
    train_tokens, test_tokens = split_train_test(tokens, config.train_percent)
    print('Loading Vocab')
    vocab = build_vocab(tokens)
    assert config.vocab_size >= len(vocab), f'vocab_size({config.vocab_size}) < len(vocab)({len(vocab)})'
    print(f"vocab size = {config.vocab_size}")
    
    print('Loading tokens_idx tensor, which may take some time')
    train_idx = torch.tensor(read_tokens_idx(train_tokens, vocab, seq_len=config.block_size), dtype=torch.int64)
    test_idx = torch.tensor(read_tokens_idx(test_tokens, vocab, seq_len=config.block_size), dtype=torch.int64)
    

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        print('cuda not available, use cpu instead')
    device = torch.device(config.device)
    
    # 使用混合精度训练
    dtype = 'bfloat16' if config.device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    print(f'dtype={dtype}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
    # 使用grad scaler，在混合精度训练中避免梯度 underflow 的问题
    # 反向传播前动态放大梯度的数值范围，然后在更新模型参数前再将梯度缩小到原始范围。
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    pad_idx = vocab['<pad>'] 

    train_data_loader = tokens_dataloader(train_idx, config.train_batch, pad_idx, is_train=True)
    test_data_loader = tokens_dataloader(test_idx, config.test_batch, pad_idx, is_train=False)
    
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
    
    optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        model, global_step, global_epoch = checkpoint_manager.load(config.resume_from, model, optimizer)
        
    if config.compile:
        print('compiling the model... this might take a while')
        model = torch.compile(model)
    
    writer = SummaryWriter(config.tensorboard_path)
    
    print('Start Training')
    while global_epoch < config.max_epoch:
        model.train()
        prog_bar = tqdm(train_data_loader)
        for x, y in prog_bar:
            x = x.to(device)
            y = y.to(device)
            global_step += 1
            lr = calc_lr(config, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            with ctx:
                _, loss = model(x, y)
            
            prog_bar.set_description(f'{global_step}|{global_epoch}|loss={loss.item():.8}')
            writer.add_scalar('loss', loss.item(), global_step)
            # loss.backward()
            scaler.scale(loss).backward()
            # 裁剪梯度，防止梯度爆炸
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            
            if global_step % config.save_freq == 0:
                checkpoint_manager.save(model, config, global_step, global_epoch, optimizer)
        
            if global_step % config.eval_freq == 0:
                print(f'Testing')
                prog_bar = tqdm(test_data_loader)
                model.eval()
                total_loss = 0
                test_step = 0
                with torch.no_grad():
                    for x, y in prog_bar:
                        x = x.to(device)
                        y = y.to(device)
                        with ctx:
                            _, loss = model(x, y)
                        total_loss += loss.item()
                        prog_bar.set_description(f'{test_step}|{global_epoch}|loss={loss.item():.8}')
                        test_step += 1
                writer.add_scalar('test_loss', total_loss/test_step, global_step)
                model.train()
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
    
    
