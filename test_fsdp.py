#!/usr/bin/env python3
"""
FSDP测试脚本
用于验证FSDP配置是否正确，模型是否能正常初始化和前向传播
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import os
import argparse
from model import GPT


def setup_distributed():
    """初始化分布式环境"""
    if 'LOCAL_RANK' not in os.environ:
        print("警告: 未检测到分布式环境，将在单GPU模式下运行")
        return False, 0, 1
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 设置CUDA设备
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    
    return True, local_rank, world_size


def get_fsdp_wrap_policy():
    """获取FSDP包装策略"""
    from model.block import DecoderBlock
    
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DecoderBlock},
    )
    return wrap_policy


def get_mixed_precision_policy():
    """获取混合精度策略"""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
    return None


def test_fsdp_model(args):
    """测试FSDP模型"""
    # 设置分布式环境
    is_distributed, local_rank, world_size = setup_distributed()
    is_main_process = local_rank == 0 if is_distributed else True
    
    if is_main_process:
        print("=" * 50)
        print("FSDP 模型测试")
        print("=" * 50)
        print(f"分布式训练: {is_distributed}")
        if is_distributed:
            print(f"Local Rank: {local_rank}")
            print(f"World Size: {world_size}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU: {torch.cuda.current_device()}")
            print(f"BF16 支持: {torch.cuda.is_bf16_supported()}")
    
    # 设置设备
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"使用设备: {device}")
    
    # 创建模型
    if is_main_process:
        print("\n创建模型...")
    
    model = GPT(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,  # 测试时禁用dropout
        bias=True,
    )
    
    # 如果是分布式环境，使用FSDP包装模型
    if is_distributed:
        if is_main_process:
            print("使用FSDP包装模型...")
        
        wrap_policy = get_fsdp_wrap_policy()
        mixed_precision_policy = get_mixed_precision_policy()
        
        # 选择分片策略
        sharding_strategy_map = {
            'FULL_SHARD': ShardingStrategy.FULL_SHARD,
            'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
            'NO_SHARD': ShardingStrategy.NO_SHARD,
            'HYBRID_SHARD': ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = sharding_strategy_map.get(args.sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        if is_main_process:
            print(f"分片策略: {args.sharding_strategy}")
            print(f"混合精度: {mixed_precision_policy is not None}")
        
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sharding_strategy=sharding_strategy,
            device_id=local_rank,
            limit_all_gathers=True,
        )
    else:
        model = model.to(device)
    
    if is_main_process:
        print("模型创建成功！")
    
    # 测试前向传播
    if is_main_process:
        print("\n测试前向传播...")
    
    batch_size = 2
    seq_len = args.block_size
    
    # 创建测试数据
    x = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    
    try:
        model.train()
        logits, loss = model(x, y)
        
        if is_main_process:
            print(f"前向传播成功！")
            print(f"输入形状: {x.shape}")
            print(f"输出形状: {logits.shape}")
            print(f"损失值: {loss.item():.6f}")
        
        # 测试反向传播
        if is_main_process:
            print("\n测试反向传播...")
        
        loss.backward()
        
        if is_main_process:
            print("反向传播成功！")
        
        # # 测试生成
        # if is_main_process:
        #     print("\n测试生成...")
        
        # model.eval()
        # with torch.no_grad():
        #     # 测试单步生成
        #     test_input = x[:1, :10]  # 取第一个样本的前10个token
        #     pred = model.generate_once(test_input, temperature=1.0, top_k=10)
            
        #     print(f"生成测试成功！")
        #     print(f"输入序列长度: {test_input.shape[1]}")
        #     print(f"生成token: {pred.item()}")
    
    except Exception as e:
        if is_main_process:
            print(f"测试失败: {str(e)}")
        return False
    
    # 内存使用统计
    if torch.cuda.is_available() and is_main_process:
        print(f"\n内存使用统计:")
        print(f"已分配内存: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"缓存内存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    if is_main_process:
        print("\n" + "=" * 50)
        print("所有测试通过！FSDP配置正常。")
        print("=" * 50)
    
    # 清理分布式环境
    if is_distributed:
        dist.destroy_process_group()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='FSDP测试脚本')
    parser.add_argument('--block_size', type=int, default=256, help='序列长度')
    parser.add_argument('--vocab_size', type=int, default=7587, help='词汇表大小')
    parser.add_argument('--n_layer', type=int, default=12, help='层数')
    parser.add_argument('--n_head', type=int, default=12, help='注意力头数')
    parser.add_argument('--n_embd', type=int, default=768, help='嵌入维度')
    parser.add_argument('--sharding_strategy', type=str, default='FULL_SHARD',
                        choices=['FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD'],
                        help='FSDP分片策略')
    
    args = parser.parse_args()
    
    # 运行测试
    success = test_fsdp_model(args)
    
    if not success:
        exit(1)


if __name__ == '__main__':
    main() 