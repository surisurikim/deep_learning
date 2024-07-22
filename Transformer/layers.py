"""
    @ author : Seul Kim
    @ when : Jan 03, 2024
    @ contact : niceonesuri@gmail.com
    @ blog : https://smartest-suri.tistory.com/
"""


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

    
# 위치 인코딩(Positional Embedding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int,
                       dropout: float,
                       maxlen: int = 5000,
                       device = None):
        super(PositionalEncoding, self).__init__()
        
        # 위치 정의 벡터
        pos = torch.arange(0, maxlen, device = device).reshape(maxlen, 1)
        # 위치에 곱해줄 값 정의
        den = torch.exp(-torch.arange(0, d_model, 2, device = device) * math.log(10000) / d_model)
        # 포지셔널 인코딩 벡터 초기값 설정 (모든 자리 0으로 시작)
        pe = torch.zeros((maxlen, d_model))
        # 포지셔널 인코딩 마지막 차원이 짝수일 때 (슬라이싱 0::2 -> 0부터 시작해서 스텝 2씩이니까 짝수)
        pe[:, 0::2] = torch.sin(pos * den) # 싸인함수
        # 포지셔널 인코딩 마지막 차원이 홀수일 때 (슬라이싱 1::2 -> 1부터 시작해서 스텝 2씩이니까 홀수)
        pe[:, 1::2] = torch.cos(pos * den) # 코싸인함수
        # 차원 추가
        pe = pe.unsqueeze(0) # 임베딩 결과값이랑 더해야되니깐 shape 맞춰주기
        
        self.dropout = nn.Dropout(dropout) # dropout 추가
        self.register_buffer('pe', pe) # pe 벡터를 buffer로 register : state_dict()에는 parameter와 buffer가 있는데, 그 중 buffer로 등록 -> 학습할때 update 되지 않도록 고정 
        
    def forward(self, x: torch.Tensor):
        seq_length = x.size(1) # 입력 시퀀스의 길이 반환
        pe = self.pe[:, :seq_length, :].expand(x.size(0), -1, -1) # 입력 시퀀스의 길이에 맞춰 위치 인코딩 텐서를 슬라이싱
        return self.dropout(x + pe)

# 워드 임베딩 -> 파이토치 nn.Embeding : https://wikidocs.net/64779
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, tokens: torch.Tensor):
        # 토큰 임베딩에 √d_model 곱해주기 (논문 3.4장에 그러랍디다)
        out = self.embedding(tokens.long()) * math.sqrt(self.d_model)
        # self.long()는 self.to(torch.int64)와 같은 역할
        return out

# "트랜스포머 임베딩" 만들어주기 (임베딩 + 포지셔널 인코딩)
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, drop_prob, max_len, device)
        
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_enc = self.pos_enc(tok_emb) # 두개 더하는건 이미 pos에서 했음
        return pos_enc

# --------------------------------------- 여기까지 잘됐는지 확인
if __name__ == "__main__":
    vocab_size = 10000
    d_model = 512
    max_len = 5000
    drop_prob = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
    input_tokens = torch.randint(0, vocab_size, (32,100)).to(device)
    output = model(input_tokens)
    print(output.shape)
# ---------------------------------------- 확인완료

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
        self.dropout = nn.Dropout(p = drop_prob)
        self.linear2 = nn.Linear(hidden, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
        

# %%
