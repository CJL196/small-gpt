import collections
import torch
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

def read_tokens(path:str):
    """读取数据
    返回一个二维列表，每个元素为token字符
    同一行的属于同一个对话，同一对话内用'sep'分隔说话者
    """
    with open(path, 'r') as f:
        data = f.read()
        tokens = data.split("\n\n")
        for i, d in enumerate(tokens):
            d = list(d)
            for j, k in enumerate(d):
                if k == '\n':
                    d[j] = '<sep>'
            tokens[i]=d
    return tokens


def read_tokens_idx(tokens, vocab, seq_len=None, show_bar=True):
    """
    获取标号数据集
    返回一个2维tensor，每个元素为token字符对应的标号
    ret.shape = (len(tokens), seq_len)
    """
    if seq_len is None:
        seq_len = max([len(token) for token in tokens])
    ret = np.zeros((len(tokens), seq_len), dtype=np.int16)
    pad_idx = vocab['<pad>'] # 大多数token为<pad>，避免重复查询vocab
    if show_bar:
        for i, sentence in tqdm(enumerate(tokens), desc='Loading tokens idx', total=len(tokens)):
            for j in range(seq_len):
                if j < len(sentence):
                    ret[i][j] = vocab[sentence[j]]
            ret[i][len(sentence):] = pad_idx
    else:
        for i, sentence in enumerate(tokens):
            for j in range(seq_len):
                if j < len(sentence):
                    ret[i][j] = vocab[sentence[j]]
            ret[i][len(sentence):] = pad_idx
    return ret

def build_vocab(tokens):
    """构建词表"""
    reserved_tokens = ['<sep>', '<pad>']
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=reserved_tokens)
    return vocab

class Vocab:  
    """文本词表"""
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
        # 将词元列表展平成一个列表
    tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def tokens_dataloader(tokens_idx, batch_size, pad_idx, is_train=True):
    """构造一个PyTorch数据迭代器"""
    from torch.utils import data
    add = torch.full((tokens_idx.shape[0], 1), pad_idx, dtype=torch.int64)
    y = torch.cat([tokens_idx[:, 1:], add], dim=-1)
    dataset = data.TensorDataset(tokens_idx, y)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == "__main__":
    path = 'data/train.txt'
    print('读取tokens')
    tokens = read_tokens(path)
    print(tokens[:2])
    v = build_vocab(tokens)
    
    print('根据标号获取字符')
    for i in range(10):
        print(v.to_tokens(i), end=' ')
    print()
    
    print('根据字符获取标号')
    print(v['乐'])
    
    print('获取token出现频率')
    print(v.token_freqs[0:20])
    
    print('获取总token数')
    print(len(v))
    
    print('获取标号数据集')
    tokens_idx = read_tokens_idx(tokens, v, 1024)
    print(tokens_idx[:2])
    print(tokens_idx.shape)
    
    print('最大句子长度')
    print(max([len(sentence) for sentence in tokens_idx]))
    
    

"""
读取tokens
[['谢', '谢', '你', '所', '做', '的', '一', '切', '<sep>', '你', '开', '心', '就', '好', '<sep>', '开', '心', '<sep>', '嗯', '因', '为', '你', '的', '心', '里', '只', '有', '学', '习', '<sep>', '某', '某', '某', '，', '还', '有', '你', '<sep>', '这', '个', '某', '某', '某', '用', '的', '好'], ['你', '们', '宿', '舍', '都', '是', '这', '么', '厉', '害', '的', '人', '吗', '<sep>', '眼', '睛', '特', '别', '搞', '笑', '这', '土', '也', '不', '好', '捏', '但', '就', '是', '觉', '得', '挺', '可', '爱', '<sep>', '特', '别', '可', '爱', '啊']]
根据标号获取字符
<unk> <sep> <pad> 我 的 ， 你 了 哈 是 
根据字符获取标号
219
获取token出现频率
[('<sep>', 1548664), ('我', 653547), ('的', 611458), ('，', 597859), ('你', 554873), ('了', 520155), ('哈', 493861), ('是', 472772), ('不', 459928), ('好', 300303), ('一', 256255), ('有', 215029), ('这', 196945), ('么', 186024), ('！', 180269), ('个', 179892), ('就', 178807), ('啊', 175506), ('看', 172787), ('没', 163021)]
获取总token数
7587
获取标号数据集
Loading tokens idx: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 500001/500001 [00:16<00:00, 31116.49it/s]
[[ 72  72   6 ...   2   2   2]
 [  6  53 863 ...   2   2   2]]
(500001, 1024)
最大句子长度
1024
"""