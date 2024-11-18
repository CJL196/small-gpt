# Small GPT

本项目参考：
- [动手学深度学习](https://zh.d2l.ai/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## 训练

你需要指定配置文件，默认可用`config/default.yaml`

```
python train.py config/default.yaml     
```

## 对话

在对话前，请修改您的配置文件如`config/default.yaml`，修改`resume_from`指定一个模型的路径

```
python chat.py config/default.yaml
```

![demo](assets/demo1.jpg)