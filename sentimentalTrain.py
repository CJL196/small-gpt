from dataParser import read_tokens, read_tokens_idx, build_vocab
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from model import GPT
import os, argparse
from dataParser import tokens_dataloader
from checkpointManager import CheckpointManager
from sentimentalData import get_data
import math
from contextlib import nullcontext

global_step = 0
global_epoch = 0

def change_model_head(model, device, num_classes=2):
    lm_head = model.lm_head
    new_lm_head = nn.Linear(lm_head.in_features, num_classes, bias=lm_head.bias).to(device)
    setattr(model, 'lm_head', new_lm_head)

    return model
    
def add_start_and_extract_to_model(model, vocab, device):
    """
    添加新的token到模型的Embedding中
    """
    original_embedding = model.transformer.wte
    new_embedding = nn.Embedding(len(vocab), original_embedding.embedding_dim).to(device)
    new_embedding.weight.data[:len(vocab)-2] = original_embedding.weight.data
    # print(new_embedding.weight)
    model.transformer.wte = new_embedding
    return model

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
    print('Loading Vocab')
    tokens = read_tokens(config.data_path)
    
    vocab = build_vocab(tokens)
    add_tokens = ['<start>', '<extract>']
    vocab.insert_token(add_tokens)
    
    assert config.vocab_size + 2 >= len(vocab), f'vocab_size({config.vocab_size + 2}) < len(vocab)({len(vocab)})'
    print(f"Vocab Loaded, size = {config.vocab_size + 2}")
    
    train_loader, test_loader = get_data(vocab, config)
    print('Sentimental train data loaded')

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        print('cuda not available, use cpu instead')
    device = torch.device(config.device)
    pad_idx = vocab['<pad>'] 
    sep_idx = vocab['<sep>']
    start_idx = vocab['<start>']
    extract_idx = vocab['<extract>']
    print(f'pad_idx={pad_idx}, sep_idx={sep_idx}, start_idx={start_idx}, extract_idx={extract_idx}')
    
    # 使用混合精度训练
    dtype = 'bfloat16' if config.device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    print(f'dtype={dtype}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
    # 使用grad scaler，在混合精度训练中避免梯度 underflow 的问题
    # 反向传播前动态放大梯度的数值范围，然后在更新模型参数前再将梯度缩小到原始范围。
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

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
    
    
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root)
    if config.resume_from is not None:
        print(f'resume from {config.resume_from}')
        model = change_model_head(model, device, num_classes=2)
        model = add_start_and_extract_to_model(model, vocab, device)
        optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
        model, _, _ = checkpoint_manager.load(config.resume_from, model, optimizer, compile=False)
    elif config.base_model is not None:
        print(f'load base model from {config.base_model}')
        model, _, _ = checkpoint_manager.load(config.base_model, model, compile=False)
        model = change_model_head(model, device, num_classes=2)
        model = add_start_and_extract_to_model(model, vocab, device)
        optimizer = model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    else:
        raise ValueError('You should specify resume_from or base_model in config file')
    
    if config.frozen:
        unfrozen_params = [f'h.{config.n_layer-1}', 'ln_f']
        for name, param in model.named_parameters():
            if any([f in name for f in unfrozen_params]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f'{name} is not frozen')
    
    if config.compile:
        print('compiling the model... this might take a while')
        model = torch.compile(model)
        
    writer = SummaryWriter(config.tensorboard_path)
    
    print('Start Training')
    while global_epoch < config.max_epoch:
        model.train()
        prog_bar = tqdm(train_loader)
        for x, y, l in prog_bar:
            # print(x.shape, y.shape, l.shape) # torch.Size([24, 256]) torch.Size([24, 1]) torch.Size([24])
            x = x.to(device)
            y = y.to(device)
            global_step += 1
            lr = calc_lr(config, global_step)
            writer.add_scalar('lr', lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.zero_grad()
            with ctx:
                logits = model(x, sentimental=True)
                if config.use_mask:
                    logits = logits[torch.arange(len(logits)), l-1, :]
                    logits = logits.view(-1, 1, logits.shape[-1])
                else:
                    logits = logits[:, [-1], :]
                # print(logits.shape, y.shape) # torch.Size([24, 1, 2]) torch.Size([24, 1])
                loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
                accuracy = (logits.argmax(dim=-1) == y).float().mean()
            prog_bar.set_description(f'{global_step}|{global_epoch}|loss={loss.item():.8}|accuracy={accuracy:.4}')
            writer.add_scalar('loss', loss.item(), global_step)
            writer.add_scalar('accuracy', accuracy, global_step)
            # loss.backward()
            scaler.scale(loss).backward()
            # 裁剪梯度，防止梯度爆炸
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            if global_step % config.save_freq == 0:
                checkpoint_manager.save(model, config, global_step, global_epoch, optimizer)
        
            if global_step % config.eval_freq == 0:
                # print(f'Testing')
                test_prog_bar = tqdm(test_loader)
                model.eval()
                total_loss = 0
                test_step = 0
                total_accuracy = 0
                with torch.no_grad():
                    for x, y, l in test_prog_bar:
                        x = x.to(device)
                        y = y.to(device)
                        with ctx:
                            logits = model(x, sentimental=True)
                            if config.use_mask:
                                logits = logits[torch.arange(len(logits)), l-1, :]
                                logits = logits.view(-1, 1, logits.shape[-1])
                            else:
                                logits = logits[:, [-1], :]
                            loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
                            accuracy = (logits.argmax(dim=-1) == y).float().mean()
                        total_loss += loss.item()
                        total_accuracy += accuracy.item()
                        test_prog_bar.set_description(f'{test_step}|{global_epoch}|loss={loss.item():.8}|accuracy={accuracy:.4}')
                        test_step += 1
                writer.add_scalar('test_loss', total_loss/test_step, global_step)
                writer.add_scalar('test_accuracy', total_accuracy/test_step, global_step)
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
    
    
