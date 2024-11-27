import pandas as pd
from dataParser import read_tokens_idx
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch

def split_train_test(tokens, labels, train_percent):
    total_dialog = len(tokens)
    train_dialog = int(total_dialog * train_percent)
    train_tokens = tokens[:train_dialog]
    test_tokens = tokens[train_dialog:]
    train_labels = labels[:train_dialog]
    test_labels = labels[train_dialog:]
    return train_tokens, test_tokens, train_labels, test_labels

def get_data(vocab, config):
    all_data = pd.read_csv(config.sentimental_data_path, dtype=str)
    # 删除评论前后空格
    all_data = all_data.applymap(lambda x: str(x).strip())
    # 打乱数据
    all_data = all_data.sample(frac=1, random_state=config.seed).reset_index(drop=True)
    
    tokens, labels = [], []
    for i in range(len(all_data)):
        review = all_data.iloc[i, 1]
        label = all_data.iloc[i, 0]
        review = list(review)
        i = 0
        # 消息足够短，剩余空间填'<pad>'
        if len(review) <= config.block_size:
            tokens.append(review + ['<pad>']*(config.block_size - len(review)))
            labels.append(label)
            continue
        # 消息足够长，分段
        while i + config.block_size < len(review):
            tokens.append(review[i:i + config.block_size])
            labels.append(label)
            i += config.block_size
        tokens.append(review[-config.block_size:])
        labels.append(label)
    
    train_tokens, test_tokens, train_labels, test_labels = split_train_test(tokens, labels, config.train_percent)
    
    train_idx = read_tokens_idx(train_tokens, vocab, show_bar=True)
    test_idx = read_tokens_idx(test_tokens, vocab, show_bar=True)
    
    train_labels = np.array(train_labels, dtype=np.int64).reshape((-1, 1))
    test_labels = np.array(test_labels, dtype=np.int64).reshape((-1, 1))
    
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    print(f'total train data: {train_idx.shape[0]}, total test data: {test_idx.shape[0]}')
    train_set = TensorDataset(train_idx, train_labels,)
    test_set = TensorDataset(test_idx, test_labels)
    
    train_loader = DataLoader(train_set, batch_size=config.train_batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.test_batch, shuffle=True)
    
    return train_loader, test_loader


    
    
    