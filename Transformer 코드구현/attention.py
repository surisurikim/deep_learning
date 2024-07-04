import torch.nn as nn
import torch
import math
import numpy as np

# 싱글 ----------------------------------------------------------
class ScaleDotProductAttention(nn.Module):
    """
        scale dot product attention - single 어텐션 계산을 수행
    """
    
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask = None, e = 1e-12):
        batch_size, head, length, d_tensor = k.size()
        
        # 1. Q, K matrix multification 계산 - 유사성 계산
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        
        # 2. decoder의 경우 Masking 작업 추가
        if mask is not None:
            score = score.masked_fill(mask == 0, -float('inf'))
        
        # 3. softmax
        score = self.softmax(score)
        
        # 4. V matrix multification 계산
        v = (score @ v)
        
        return v, score


# 멀티 ----------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 512, n_heads = 8):
        """
        입력 - embed_dim : 임베딩 벡터의 아웃풋 차원
              n_heads : self attention이 몇개의 헤드를 사용할건지 (n_heads = 8인 경우 하나의 어탠션 헤드에서 64개의 차원 사용)
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 512
        self.n_heads = n_heads # 8
        self.attention = ScaleDotProductAttention() ########## ***
        
        self.w_q = nn.Linear(d_model, d_model, bias = False) # q, k, v의 weight map 생성 - 나중에 train 모드에서 학습됨
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_concat = nn.Linear(d_model, d_model, bias = False) # 마지막 concat 하고나서 linear 해줄 부분의 weight map
        
    
    def split(self, tensor):
        """
            n_heads로 입력받은 tensor를 split
        """
        batch_size, length, d_model = tensor.size() # (32, 10, 512)

        d_tensor = d_model // self.n_head # 512 // 8 = 64
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2) # 차원 재구성 (reshape) -> (32, 10, 8, 64) -> (32, 8, 10, 64)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
            self.split() 함수를 통해 멀티헤드로 난도질한 함수를 다시 이어붙임
        """    
        
        batch_size, n_heads, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2) # split에서 transpose 했던거 다시 원상복귀
        tensor = tensor.contiguous().view(batch_size, length, self.d_model) # (32, 10, 512) / # 텐서를 contiguous = True 상태로 변경(transpose 하면서 저장상태가 바뀌었을 경우...)
        
        
    def forward(self, q, k, v, mask = None):
        # 1. Q, K, V 생성 (weight map이랑 dot product 계산)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. Q, K, V 나누기 (multi-head로)
        q, k, v = self.split(q), self.split(k), self.split(v)
        
        # 3. single_attention 계산
        out, attention = attention(q, k, v, mask = mask) ########## ***
        
        # 4. concatenate - 이어 붙이기!
        out = self.concat(out)
        out = self.w_concat(out) # 마지막 linear!
        
        return out