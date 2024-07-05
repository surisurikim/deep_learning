#%%

import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
print(torch.__version__)

# 워드 임베딩 -> 파이토치 nn.Embeding : https://wikidocs.net/64779
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        입력 - vocab_size : vocabulary의 사이즈
              d_model : embeddings의 차원(dimension) - 512...
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        """
        입력 - x : input vector
        출력 - embedding vector
        """
        out = self.embed(x)
        return out
    
# 위치 인코딩(Positional Embedding)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int,
                       dropout: float,
                       maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # 위치 정의
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        # 위치에 곱해줄 값 정의
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 포지셔널 인코딩 벡터 초기값 설정 (모든 자리 0으로 시작)
        pe = torch.zeros((maxlen, emb_size))
        # 포지셔널 인코딩 마지막 차원이 짝수일 때 (슬라이싱 0::2 -> 0부터 시작해서 스텝 2씩이니까 짝수)
        pe[:, 0::2] = torch.sin(pos * den) # 싸인함수
        # 포지셔널 인코딩 마지막 차원이 홀수일 때 (슬라이싱 1::2 -> 1부터 시작해서 스텝 2씩이니까 홀수)
        pe[:, 1::2] = torch.cos(pos * den) # 코싸인함수
        # 차원 추가
        pe = pe.unsqueeze(-2) # 해주는 이유 : input shape이 [batch_size, seq_len, d_model] 이기 때문이다!! (임베딩 결과값이랑 더해야되니깐 shape 맞춰주는거임)
        
        self.droupout = nn.Dropout(dropout) # dropout 추가
        self.register_buffer('pe', pe) # pe 벡터를 buffer로 register : state_dict()에는 parameter와 buffer가 있는데, 그 중 buffer로 등록 -> 학습할때 update 되지 않도록 고정 
        
    def forward(self, embedding: torch.Tensor):  
        pe = self.pe[:, :embedding.size(0), :]
        embedding = embedding + pe
        return self.droupout(embedding)



# 임베딩 + 포지셔널 인코딩 -> "트랜스포머 임베딩" 만들어주기
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding()





# 정규화 (나중에 EncoderLayer class에서 Add & Norm을 위해 쓰일것)
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # gamma라는 학습 가능한 파라미터 - 초기값은 1로 설정된 길이 d_model의 벡터
        self.beta = nn.Parameter(torch.zeros(d_model)) # beta라는 학습 가능한 파라미터 - 초기값은 0으로 설정된 길이 d_model의 벡터
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True) # 입력 텐서 x의 마지막 차원(-1)에 대한 평균을 계산 (예 : (32, 10, 512) -> 계산 후 (32, 10, 1))
        # keepdim = True : 결과 텐서가 원래의 차원을 유지
        var = x.var(-1, unbiased = False, keepdim = True) # 입력 텐서 x의 마지막 차원(-1)에 대한 분산을 계산
        # unbiased = False : 모집단 분산 (전체 데이터 분산을 계산 - 차원별로 분산 계산할때 자유도를 고려하지 않음! 모집단 분산이 표준화 돼있음)
        
        out = (x - mean) / torch.sqrt(var + self.eps) # 평균과 분산을 이용해서 정규화!
        out = self.gamma * out + self.beta
        return out

# 포지션와이드 피드포워드!
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(p = drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
        

# %%
