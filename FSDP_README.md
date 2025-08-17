# FSDP (Fully Sharded Data Parallel) 使用指南

## 简介

FSDP（Fully Sharded Data Parallel）是PyTorch提供的高级分布式训练技术，相比传统的DDP，它可以：

- **将模型参数、梯度和优化器状态分片到多个GPU上**
- **支持训练更大的模型**（突破单卡内存限制）
- **提供更高效的内存使用**
- **支持混合精度训练**

## 主要特性

### 与DDP的对比

| 特性 | DDP | FSDP |
|------|-----|------|
| 模型复制 | 每个GPU保存完整模型 | 模型参数分片存储 |
| 内存使用 | 每个GPU需要完整模型内存 | 内存使用与GPU数量成反比 |
| 支持模型大小 | 受限于单GPU内存 | 可训练超大模型 |
| 通信开销 | 只同步梯度 | 需要参数聚合/分散 |
| 适用场景 | 中小型模型 | 大型模型，内存受限场景 |

## 安装要求

确保您的环境满足以下要求：

```bash
# PyTorch >= 1.12.0 (推荐 >= 2.0.0)
pip install torch>=2.0.0 torchvision

# 其他依赖
pip install -r requirements.txt
```

## 配置文件

项目提供了FSDP专用的配置文件：

- `config/train90M_fsdp.yaml` - 90M参数模型的FSDP配置
- `config/train300M_fsdp.yaml` - 300M参数模型的FSDP配置

### 关键配置参数

```yaml
# FSDP配置
fsdp: True  # 启用FSDP
sharding_strategy: FULL_SHARD  # 分片策略
world_size: 4  # GPU数量
compile: False  # FSDP模式下建议禁用compile
```

### 分片策略说明

- **FULL_SHARD**: 完全分片（推荐）- 参数、梯度、优化器状态全部分片
- **SHARD_GRAD_OP**: 只分片梯度和优化器状态 - 参数仍然复制
- **NO_SHARD**: 不分片（类似DDP）- 主要用于调试
- **HYBRID_SHARD**: 混合分片 - 在节点间完全分片，节点内复制

## 使用方法

### 单节点多GPU训练

#### 2张GPU训练300M模型

```bash
torchrun --nproc_per_node=2 train_fsdp.py config/train300M_fsdp.yaml
```

#### 4张GPU训练300M模型

```bash
torchrun --nproc_per_node=4 train_fsdp.py config/train300M_fsdp.yaml
```

#### 8张GPU训练300M模型

```bash
torchrun --nproc_per_node=8 train_fsdp.py config/train300M_fsdp.yaml
```

### 多节点分布式训练

#### 2台机器，每台4张GPU（总共8张GPU）

**主节点（机器1）运行：**
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train_fsdp.py config/train300M_fsdp.yaml
```

**从节点（机器2）运行：**
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train_fsdp.py config/train300M_fsdp.yaml
```

### 参数说明

- `--nproc_per_node`: 每个节点的GPU数量
- `--nnodes`: 总节点数
- `--node_rank`: 当前节点的排名（从0开始）
- `--master_addr`: 主节点的IP地址
- `--master_port`: 通信端口

## Checkpoint管理

FSDP使用专门的checkpoint管理器来处理分片模型的保存和加载。

### 保存策略

项目提供两种保存策略：

#### 1. 完整状态字典保存（推荐）
- 所有参数聚合到rank 0保存
- 可以在不同GPU数量下加载
- 文件名：`fsdp_cpt{step}.pth`

#### 2. 分片状态字典保存
- 每个GPU保存自己的分片
- 加载时需要相同的GPU数量
- 文件名：`fsdp_sharded_cpt{step}_rank{rank}.pth`

### 恢复训练

```bash
# 从checkpoint恢复训练
torchrun --nproc_per_node=4 train_fsdp.py config/train300M_fsdp.yaml --resume_from checkpoints/fsdp_cpt4000.pth
```

## 内存优化建议

### 1. 选择合适的分片策略

```yaml
# 对于大模型，使用完全分片
sharding_strategy: FULL_SHARD

# 对于中等模型，可以只分片梯度
sharding_strategy: SHARD_GRAD_OP
```

### 2. 启用混合精度训练

FSDP自动支持混合精度，会根据GPU能力选择：
- 支持BF16的GPU：使用bfloat16
- 其他GPU：使用float16

### 3. 调整批次大小

```yaml
# FSDP可以支持更大的总批次大小
train_batch: 64  # 总批次大小，自动分配到各GPU
```

### 4. 模型包装策略

FSDP会自动包装`DecoderBlock`层，实现最优的内存分片。

## 性能监控

### 内存使用监控

```bash
# 监控GPU内存使用
nvidia-smi -l 1

# 或使用gpustat
gpustat -i 1
```

### 训练指标

- 训练过程中会显示每个step的loss
- Tensorboard日志保存在`./tensorboard`目录
- 只有rank 0会记录日志，避免重复

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   RuntimeError: CUDA out of memory
   ```
   解决方案：
   - 减小batch_size
   - 使用FULL_SHARD策略
   - 增加GPU数量

2. **分布式通信错误**
   ```bash
   RuntimeError: NCCL error
   ```
   解决方案：
   - 检查网络连接
   - 确保防火墙允许指定端口
   - 验证所有节点可以相互访问

3. **Checkpoint加载失败**
   ```bash
   RuntimeError: Error loading checkpoint
   ```
   解决方案：
   - 确保checkpoint文件完整
   - 检查模型配置是否匹配
   - 验证PyTorch版本兼容性

### 调试技巧

1. **启用详细日志**
   ```bash
   export TORCH_DISTRIBUTED_DEBUG=INFO
   ```

2. **单GPU测试**
   ```bash
   # 先在单GPU上测试模型是否正常
   python train_fsdp.py config/train300M_fsdp.yaml
   ```

3. **检查模型分片**
   FSDP会自动打印分片信息，查看日志确认分片正确。

## 性能优化建议

### 1. 网络优化
- 使用InfiniBand网络（如果可用）
- 确保节点间网络带宽充足
- 考虑网络拓扑结构

### 2. 数据加载优化
- 使用多进程数据加载
- 预先处理数据并缓存
- 考虑使用更快的存储（SSD/NVMe）

### 3. 模型优化
- 合理设置block大小
- 调整模型层数和宽度
- 使用适当的dropout率

## 示例脚本

### 快速开始脚本

```bash
#!/bin/bash
# fsdp_train.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG=INFO

# 启动FSDP训练
torchrun --nproc_per_node=4 \
    train_fsdp.py config/train300M_fsdp.yaml
```

### 多机训练脚本

```bash
#!/bin/bash
# fsdp_multi_node.sh

MASTER_ADDR=${1:-"192.168.1.100"}
NODE_RANK=${2:-0}
NNODES=${3:-2}

torchrun --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --nproc_per_node=8 \
         --master_addr=$MASTER_ADDR \
         --master_port=29500 \
         train_fsdp.py config/train300M_fsdp.yaml
```

## 总结

FSDP为训练大型语言模型提供了强大的支持，通过合理配置可以：

- 突破单GPU内存限制
- 实现线性扩展的训练速度
- 高效利用多GPU资源
- 支持超大模型训练

根据您的硬件配置和模型大小，选择合适的分片策略和配置参数，即可充分发挥FSDP的优势。 