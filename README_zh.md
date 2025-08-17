# Small GPT

本项目训练了一个小型GPT模型，能够进行简单的对话，并微调用于情感分类

## 准备数据集

数据集存放在 data 文件夹下

中文闲聊对话`data/train.txt`

https://drive.google.com/file/d/1nEuew_KNpTMbyy7BO4c8bXMXN351RCPp/view

情感分类`data/ChnSentiCorp_htl_all.csv`

https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv 

## 环境配置

```bash
pip install -r requirements.txt
```

## 预训练

### 单GPU训练

模型参数量90M

```
python train.py config/train90M.yaml     
```

模型参数量300M

```
python train.py config/train300M.yaml     
```

默认配置训练需要约16GB显存，你可以根据实际的硬件条件修改batch size

### 分布式训练（多GPU）

启用分布式训练可以显著加速训练过程。项目已支持 DistributedDataParallel (DDP)。

#### 单节点多GPU训练

使用2个GPU训练300M模型：

```bash
torchrun --nproc_per_node=2 train.py config/train300M_ddp.yaml
```

使用4个GPU训练300M模型：

```bash
torchrun --nproc_per_node=4 train.py config/train300M_ddp.yaml
```

使用8个GPU训练300M模型：

```bash
torchrun --nproc_per_node=8 train.py config/train300M_ddp.yaml
```

使用2个GPU训练90M模型：

```bash
torchrun --nproc_per_node=2 train.py config/train90M_ddp.yaml
```

#### 多节点分布式训练（跨机器）

如果你有多台机器，可以进行跨节点的分布式训练以获得更大的加速比。

**示例：2台机器，每台8个GPU（总共16个GPU）**

在**主节点**（节点0）上运行：
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train.py config/train300M_ddp.yaml
```

在**从节点**（节点1）上运行：
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train.py config/train300M_ddp.yaml
```

**参数说明：**
- `--nnodes=2`: 总共2个节点（机器）
- `--node_rank=0/1`: 当前节点编号（主节点为0）
- `--nproc_per_node=8`: 每个节点使用的GPU数量
- `--master_addr`: 主节点的IP地址
- `--master_port`: 通信端口（确保端口未被占用）

**环境要求：**
- 所有节点必须能够通过网络相互访问
- 所有节点必须有相同的代码和数据
- 防火墙允许指定端口的通信
- 建议使用高速网络（如InfiniBand）以获得最佳性能

**注意事项：**
- 分布式训练会自动在多个GPU之间分配batch size
- 只有主进程（rank 0）会保存模型和写入tensorboard日志
- 确保所有GPU都有足够的显存
- 分布式配置文件中的 `train_batch` 是总batch size，会被自动分配到各个GPU

## 预训练模型🤗

你可以在hugging face🤗上下载预训练模型

```bash
mkdir -p checkpoints/pretrained
cd checkpoints/pretrained
wget https://huggingface.co/cjl196/small-gpt/resolve/main/cpt90M.pth -O cpt90M.pth
wget https://huggingface.co/cjl196/small-gpt/resolve/main/cpt300M.pth -O cpt300M.pth
```

## 对话

如果你希望使用自己训练的模型，在对话前，请修改配置文件中`resume_from`的值为模型的路径

使用下面的指令，和预训练的300M模型对话

```bash
python chat.py config/chat300M.yaml
```

## 对话效果

![demo](assets/demo1.png)
![demo](assets/demo2.png)
![demo](assets/demo3.png)
![demo](assets/demo4.png)

## 情感分类

基于预训练的300M模型，训练情感分类器

情感分类提供多个配置文件`config/sentimental*.yaml`，主要区别是是否mask、是否冻结参数，可用于消融实验

```bash
# mask&not_frozen
python sentimentalTrain.py config/sentimental.yaml
# mask&frozen
python sentimentalTrain.py config/sentimental1.yaml
# no_mask&not_frozen
python sentimentalTrain.py config/sentimental2.yaml
```

消融实验效果：

|                   | 准确度    | 训练时间     |
| ----------------- | --------- | ------------ |
| 无mask&无冻结参数 | **91.3%** | 1hr          |
| 有mask&无冻结参数 | **91.2%** | 1hr          |
| 有mask&有冻结参数 | 87.8%     | **26.63min** |

## 致谢

本项目参考以下仓库或教程，在此特别鸣谢

- [动手学深度学习](https://zh.d2l.ai/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)