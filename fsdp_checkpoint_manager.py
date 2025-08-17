import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType


class FSDPCheckpointManager:
    """
    FSDP专用的Checkpoint管理器
    处理FSDP模型的特殊保存和加载逻辑
    """
    def __init__(self, save_root):
        self.save_root = save_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    
    def save(self, model, config, global_step, global_epoch, optimizer, is_main_process=True):
        """
        保存FSDP模型checkpoint
        
        Args:
            model: FSDP包装的模型
            config: 训练配置
            global_step: 全局步数
            global_epoch: 全局epoch
            optimizer: 优化器
            is_main_process: 是否为主进程
        """
        if not is_main_process:
            return
            
        filename = f'fsdp_cpt{global_step}.pth'
        path = os.path.join(self.save_root, filename)
        
        # 使用FSDP的完整状态字典配置
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
        
        # 只在rank 0保存
        if dist.get_rank() == 0:
            checkpoint = {
                'global_step': global_step,
                'global_epoch': global_epoch,
                'block_size': config.block_size,
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd,
                'bias': config.bias,
                'dataset': config.data_path,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'fsdp': True,  # 标记为FSDP checkpoint
                'sharding_strategy': getattr(config, 'sharding_strategy', 'FULL_SHARD'),
            }
            torch.save(checkpoint, path)
            print(f'FSDP Checkpoint saved: {path}')
    
    def load(self, path, model, optimizer=None):
        """
        加载FSDP模型checkpoint
        
        Args:
            path: checkpoint文件路径
            model: FSDP包装的模型
            optimizer: 优化器（可选）
            
        Returns:
            tuple: (model, global_step, global_epoch)
        """
        print(f'Loading FSDP State Dict from {path}')
        
        # 只在rank 0加载checkpoint
        if dist.get_rank() == 0:
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = None
        
        # 广播checkpoint信息到所有ranks
        if dist.is_initialized():
            if dist.get_rank() == 0:
                global_step = checkpoint['global_step']
                global_epoch = checkpoint['global_epoch']
            else:
                global_step = None
                global_epoch = None
            
            # 广播步数和epoch信息
            global_step_tensor = torch.tensor(global_step if global_step is not None else 0)
            global_epoch_tensor = torch.tensor(global_epoch if global_epoch is not None else 0)
            dist.broadcast(global_step_tensor, src=0)
            dist.broadcast(global_epoch_tensor, src=0)
            global_step = global_step_tensor.item()
            global_epoch = global_epoch_tensor.item()
        else:
            global_step = checkpoint['global_step']
            global_epoch = checkpoint['global_epoch']
        
        # 加载模型状态字典
        if dist.get_rank() == 0:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = None
        
        # 使用FSDP的完整状态字典配置进行加载
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            if dist.get_rank() == 0:
                model.load_state_dict(state_dict)
        
        # 同步所有进程
        if dist.is_initialized():
            dist.barrier()
        
        # 加载优化器状态（仅在rank 0）
        if optimizer is not None and dist.get_rank() == 0 and checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 打印配置信息（仅在rank 0）
        if dist.get_rank() == 0 and checkpoint is not None:
            print('FSDP checkpoint config:')
            for k, v in checkpoint.items():
                if k in ['state_dict', 'optimizer']:
                    continue
                print(f'{k}={v}')
            print('-' * 6)
        
        return model, global_step, global_epoch
    
    def save_sharded_checkpoint(self, model, config, global_step, global_epoch, optimizer):
        """
        保存分片checkpoint（每个rank保存自己的部分）
        这种方式保存更快，但需要相同数量的GPU来加载
        """
        filename = f'fsdp_sharded_cpt{global_step}_rank{dist.get_rank()}.pth'
        path = os.path.join(self.save_root, filename)
        
        # 使用LOCAL状态字典保存分片模型
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'global_step': global_step,
            'global_epoch': global_epoch,
            'rank': dist.get_rank(),
            'world_size': dist.get_world_size(),
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'bias': config.bias,
            'dataset': config.data_path,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            'fsdp_sharded': True,
            'sharding_strategy': getattr(config, 'sharding_strategy', 'FULL_SHARD'),
        }
        
        torch.save(checkpoint, path)
        print(f'FSDP Sharded Checkpoint saved: {path}')
    
    def load_sharded_checkpoint(self, path_pattern, model, optimizer=None):
        """
        加载分片checkpoint
        
        Args:
            path_pattern: checkpoint文件路径模式，例如 "checkpoints/fsdp_sharded_cpt1000_rank{}.pth"
            model: FSDP包装的模型
            optimizer: 优化器（可选）
        """
        rank = dist.get_rank()
        path = path_pattern.format(rank)
        
        print(f'Loading FSDP Sharded State Dict from {path}')
        checkpoint = torch.load(path, map_location=f'cuda:{rank}')
        
        # 验证world_size匹配
        if checkpoint['world_size'] != dist.get_world_size():
            raise ValueError(f"Checkpoint world_size ({checkpoint['world_size']}) != current world_size ({dist.get_world_size()})")
        
        # 加载模型状态字典
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model.load_state_dict(checkpoint['state_dict'])
        
        # 加载优化器状态
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        global_step = checkpoint['global_step']
        global_epoch = checkpoint['global_epoch']
        
        # 同步所有进程
        dist.barrier()
        
        if rank == 0:
            print('FSDP sharded checkpoint config:')
            for k, v in checkpoint.items():
                if k in ['state_dict', 'optimizer']:
                    continue
                print(f'{k}={v}')
            print('-' * 6)
        
        return model, global_step, global_epoch 