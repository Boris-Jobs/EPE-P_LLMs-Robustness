import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(
            hidden_size, hidden_size
        )  # 全连接层，输入输出均为隐藏层大小
        self.activation = nn.Tanh()  # Tanh激活函数应用于全连接层输出

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  # 通常第一个Token是[CLS] Token
        pooled_output = self.dense(first_token_tensor)  # 将第一个Token进行线性变换
        pooled_output = self.activation(pooled_output)  # 再用Tanh进行激活
        return pooled_output  #

    """
    hidden states的大小通常为(batch_size, sequence_length, hidden_size)
    # (num of samples, num of tokens, len of tokens)

    hidden_states = [
        [token1_sample1, token2_sample1, token3_sample1, token4_sample1, token5_sample1],
        [token1_sample2, token2_sample2, token3_sample2, token4_sample2, token5_sample2]
    ]

    hidden_states[:, 0] = [ token1_sample1, token1_sample2]
    """


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
