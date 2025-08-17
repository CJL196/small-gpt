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
# 添加分布式训练相关导入
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

global_step = 0
global_epoch = 0

def setup_distributed(config):
    """初始化分布式训练环境"""
    if config.ddp:
        # 从环境变量获取分布式训练参数
        config.local_rank = int(os.environ.get('LOCAL_RANK', config.local_rank))
        config.world_size = int(os.environ.get('WORLD_SIZE', config.world_size))
        
        # 设置当前GPU设备
        torch.cuda.set_device(config.local_rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',  # 使用NCCL后端
            init_method='env://',  # 使用环境变量初始化
            world_size=config.world_size,
            rank=config.local_rank
        )
        
        print(f"分布式训练已初始化 - Rank: {config.local_rank}, World Size: {config.world_size}")
    
    return config.local_rank == 0 or not config.ddp  # 是否为主进程

def cleanup_distributed(config):
    """清理分布式训练环境"""
    if config.ddp:
        dist.destroy_process_group()

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

    # 初始化分布式训练
    is_main_process = setup_distributed(config)
    
    # 设置随机种子（考虑分布式训练）
    seed = config.seed + (config.local_rank if config.ddp else 0)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 准备数据集
    if is_main_process:
        print('Loading tokens')
    tokens = read_tokens(config.data_path)
    # 划分训练和测试
    train_tokens, test_tokens = split_train_test(tokens, config.train_percent)
    if is_main_process:
        print('Loading Vocab')
    vocab = build_vocab(tokens)
    assert config.vocab_size >= len(vocab), f'vocab_size({config.vocab_size}) < len(vocab)({len(vocab)})'
    if is_main_process:
        print(f"vocab size = {config.vocab_size}")
    
    if is_main_process:
        print('Loading tokens_idx tensor, which may take some time')
    train_idx = torch.tensor(read_tokens_idx(train_tokens, vocab, seq_len=config.block_size), dtype=torch.int64)
    test_idx = torch.tensor(read_tokens_idx(test_tokens, vocab, seq_len=config.block_size), dtype=torch.int64)
    

    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        if is_main_process:
            print('cuda not available, use cpu instead')
    
    # 设置设备
    if config.ddp:
        device = torch.device(f'cuda:{config.local_rank}')
    else:
        device = torch.device(config.device)
    
    # 使用混合精度训练
    dtype = 'bfloat16' if config.device == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    if is_main_process:
        print(f'dtype={dtype}')
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
    # 使用grad scaler，在混合精度训练中避免梯度 underflow 的问题
    # 反向传播前动态放大梯度的数值范围，然后在更新模型参数前再将梯度缩小到原始范围。
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    pad_idx = vocab['<pad>'] 

    # 创建数据加载器，支持分布式采样
    if config.ddp:
        train_sampler = DistributedSampler(
            train_idx, 
            num_replicas=config.world_size, 
            rank=config.local_rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_idx, 
            num_replicas=config.world_size, 
            rank=config.local_rank,
            shuffle=True
        )
        # 调整batch size（每个进程的batch size）
        train_batch_size = config.train_batch // config.world_size
        test_batch_size = config.test_batch // config.world_size
    else:
        train_sampler = None
        test_sampler = None
        train_batch_size = config.train_batch
        test_batch_size = config.test_batch

    train_data_loader = tokens_dataloader(train_idx, train_batch_size, pad_idx, shuffle=(train_sampler is None), sampler=train_sampler)
    test_data_loader = tokens_dataloader(test_idx, test_batch_size, pad_idx, shuffle=(test_sampler is None), sampler=test_sampler)
    
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
    
    # 使用DDP包装模型
    if config.ddp:
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
        # 获取原始模型用于优化器配置
        raw_model = model.module
    else:
        raw_model = model
    
    optimizer = raw_model.configure_optimizers(config.weight_decay, config.lr, (config.beta1, config.beta2), config.device)
    
    if config.compile and not config.ddp:  # DDP模式下暂时不使用compile
        if is_main_process:
            print('compiling the model... this might take a while')
        model = torch.compile(model)
    
    # 加载checkpoint manager
    checkpoint_manager = CheckpointManager(config.checkpoint_root) if is_main_process else None
    if config.resume_from is not None and is_main_process:
        model, global_step, global_epoch = checkpoint_manager.load(config.resume_from, raw_model, optimizer)
        # 在分布式训练中同步全局步数和epoch
        if config.ddp:
            # 广播global_step和global_epoch到所有进程
            global_step_tensor = torch.tensor(global_step, device=device)
            global_epoch_tensor = torch.tensor(global_epoch, device=device)
            dist.broadcast(global_step_tensor, src=0)
            dist.broadcast(global_epoch_tensor, src=0)
            global_step = global_step_tensor.item()
            global_epoch = global_epoch_tensor.item()
    
    # 只在主进程创建tensorboard writer
    writer = SummaryWriter(config.tensorboard_path) if is_main_process else None
    
    if is_main_process:
        print('Start Training')
    
    try:
        while global_epoch < config.max_epoch:
            # 在分布式训练中设置epoch，确保数据shuffle的随机性
            if config.ddp:
                train_sampler.set_epoch(global_epoch)
                test_sampler.set_epoch(global_epoch)
            
            model.train()
            prog_bar = tqdm(train_data_loader) if is_main_process else train_data_loader
            for x, y in prog_bar:
                x = x.to(device)
                y = y.to(device)
                global_step += 1
                lr = calc_lr(config, global_step)
                
                # 只在主进程记录tensorboard
                if is_main_process and writer:
                    writer.add_scalar('lr', lr, global_step)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.zero_grad()
                with ctx:
                    _, loss = model(x, y)
                
                if is_main_process and hasattr(prog_bar, 'set_description'):
                    prog_bar.set_description(f'{global_step}|{global_epoch}|loss={loss.item():.8}')
                
                if is_main_process and writer:
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
                
                # 只在主进程保存模型
                if global_step % config.save_freq == 0 and is_main_process and checkpoint_manager:
                    checkpoint_manager.save(raw_model, config, global_step, global_epoch, optimizer)
            
                if global_step % config.eval_freq == 0:
                    if is_main_process:
                        print(f'Testing')
                    test_prog_bar = tqdm(test_data_loader) if is_main_process else test_data_loader
                    model.eval()
                    total_loss = 0
                    test_step = 0
                    with torch.no_grad():
                        for x, y in test_prog_bar:
                            x = x.to(device)
                            y = y.to(device)
                            with ctx:
                                _, loss = model(x, y)
                            total_loss += loss.item()
                            if is_main_process and hasattr(test_prog_bar, 'set_description'):
                                test_prog_bar.set_description(f'{test_step}|{global_epoch}|loss={loss.item():.8}')
                            test_step += 1
                            if test_step > config.eval_steps:
                                break
                    
                    # 在分布式训练中同步测试损失
                    if config.ddp:
                        total_loss_tensor = torch.tensor(total_loss, device=device)
                        test_step_tensor = torch.tensor(test_step, device=device)
                        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                        dist.all_reduce(test_step_tensor, op=dist.ReduceOp.SUM)
                        avg_test_loss = total_loss_tensor.item() / test_step_tensor.item()
                    else:
                        avg_test_loss = total_loss / test_step
                    
                    if is_main_process and writer:
                        writer.add_scalar('test_loss', avg_test_loss, global_step)
                    model.train()
            global_epoch += 1
    
    finally:
        # 清理分布式训练环境
        cleanup_distributed(config)
        if is_main_process and writer:
            writer.close()
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the config file")
    # 添加分布式训练相关参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    # 从 YAML 文件加载配置
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)
    # 从命令行参数覆盖配置
    if args.local_rank != -1:
        config.local_rank = args.local_rank
        config.ddp = True
    return config


if __name__ == '__main__':
    config = parse_args()
    main(config)
    
    
