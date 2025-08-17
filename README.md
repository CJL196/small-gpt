# Small GPT  

[中文README](./README_zh.md)

This project trains a small GPT model capable of simple conversations and fine-tuning for sentiment classification.  

## Preparing the Dataset  

The datasets are stored in the `data` folder.  

- **Chinese Chat Dataset**: `data/train.txt`  
  [Download](https://drive.google.com/file/d/1nEuew_KNpTMbyy7BO4c8bXMXN351RCPp/view)  

- **Sentiment Classification Dataset**: `data/ChnSentiCorp_htl_all.csv`  
  [Download](https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv)  

## Environment Setup  

Install the required dependencies:  

```bash
pip install -r requirements.txt
```  

## Pretraining  

### Single GPU Training

- **90M parameter model**:  

  ```bash
  python train.py config/train90M.yaml     
  ```  

- **300M parameter model**:  

  ```bash
  python train.py config/train300M.yaml     
  ```  
By default, training requires approximately **16GB of VRAM**. You can adjust the batch size based on your hardware resources.  

### Distributed Training (Multi-GPU)

Enabling distributed training can significantly accelerate the training process. The project supports two distributed training methods:
- **DDP (DistributedDataParallel)**: Traditional distributed training, suitable for most scenarios
- **FSDP (Fully Sharded Data Parallel)**: Advanced distributed training, supports larger models by sharding parameters across GPUs

#### Single-Node Multi-GPU Training

Train 300M model with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py config/train300M_ddp.yaml
```

Train 300M model with 4 GPUs:

```bash
torchrun --nproc_per_node=4 train.py config/train300M_ddp.yaml
```

Train 300M model with 8 GPUs:

```bash
torchrun --nproc_per_node=8 train.py config/train300M_ddp.yaml
```

Train 90M model with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train.py config/train90M_ddp.yaml
```

#### Multi-Node Distributed Training (Cross-Machine)

If you have multiple machines, you can perform cross-node distributed training for greater acceleration.

**Example: 2 machines, 8 GPUs each (total 16 GPUs)**

Run on **master node** (node 0):
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train.py config/train300M_ddp.yaml
```

Run on **worker node** (node 1):
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train.py config/train300M_ddp.yaml
```

**Parameter Descriptions:**
- `--nnodes=2`: Total number of nodes (machines)
- `--node_rank=0/1`: Current node ID (master node is 0)
- `--nproc_per_node=8`: Number of GPUs per node
- `--master_addr`: IP address of the master node
- `--master_port`: Communication port (ensure port is not in use)

**Environment Requirements:**
- All nodes must be able to access each other through the network
- All nodes must have the same code and data
- Firewall must allow communication on the specified port
- High-speed network (such as InfiniBand) is recommended for optimal performance

**Important Notes:**
- Distributed training automatically distributes batch size across multiple GPUs
- Only the master process (rank 0) saves models and writes tensorboard logs
- Ensure all GPUs have sufficient VRAM
- The `train_batch` in distributed config files is the total batch size, automatically distributed across GPUs

#### FSDP Training (Advanced)

FSDP enables training larger models by sharding parameters across GPUs. For detailed usage, see [FSDP_README.md](./FSDP_README.md).

**Single-Node FSDP Training:**

Train 300M model with 2 GPUs using FSDP:
```bash
torchrun --nproc_per_node=2 train_fsdp.py config/train300M_fsdp.yaml
```

Train 300M model with 4 GPUs using FSDP:
```bash
torchrun --nproc_per_node=4 train_fsdp.py config/train300M_fsdp.yaml
```

**Multi-Node FSDP Training:**

Master node:
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train_fsdp.py config/train300M_fsdp.yaml
```

Worker node:
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
  --master_addr=192.168.1.100 --master_port=29500 \
  train_fsdp.py config/train300M_fsdp.yaml
```


## Pretrained Models 🤗  

You can download the pretrained models from Hugging Face 🤗:  

```bash
mkdir -p checkpoints/pretrained
cd checkpoints/pretrained
wget https://huggingface.co/cjl196/small-gpt/resolve/main/cpt90M.pth -O cpt90M.pth
wget https://huggingface.co/cjl196/small-gpt/resolve/main/cpt300M.pth -O cpt300M.pth
```  

## Chatting  

If you want to use your own trained model, modify the `resume_from` field in the configuration file to point to your model's path.  

To chat with the pretrained **300M model**, run:  

```bash
python chat.py config/chat300M.yaml
```  

## Chat Examples  

![demo](assets/demo1.png)  
![demo](assets/demo2.png)  
![demo](assets/demo3.png)  
![demo](assets/demo4.png)  

## Sentiment Classification  

Fine-tune a sentiment classifier based on the pretrained **300M model**.  

Several configuration files (`config/sentimental*.yaml`) are provided, differing in whether masking is applied and whether parameters are frozen, allowing for ablation experiments.  

```bash
# mask & not frozen
python sentimentalTrain.py config/sentimental.yaml
# mask & frozen
python sentimentalTrain.py config/sentimental1.yaml
# no mask & not frozen
python sentimentalTrain.py config/sentimental2.yaml
```  

### Ablation Study Results  

| Configuration      | Accuracy  | Training Time  |  
| ----------------- | --------- | -------------- |  
| No mask & not frozen | **91.3%** | 1hr           |  
| Mask & not frozen | **91.2%** | 1hr           |  
| Mask & frozen    | 87.8%     | **26.63min**   |  

## Acknowledgments  

This project is inspired by the following repositories and tutorials. Special thanks to:  

- [Dive into Deep Learning](https://zh.d2l.ai/)  
- [nanoGPT](https://github.com/karpathy/nanoGPT)  