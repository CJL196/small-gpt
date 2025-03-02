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

## Training  

- **90M parameter model**:  

  ```bash
  python train.py config/train90M.yaml     
  ```  

- **300M parameter model**:  

  ```bash
  python train.py config/train300M.yaml     
  ```  

By default, training requires approximately **16GB of VRAM**. You can adjust the batch size based on your hardware resources.  

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