from layers import Embedding, PositionalEncoding, PositionwiseFeedForward, LayerNorm
from attention import MultiHeadAttention
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, n_heads = n_heads) 
        self.norm1 = LayerNorm(d_model = d_model) 
        self.dropout1 = nn.Dropout(p = drop_prob) 
        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model = d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)
        
    def forward(self, x, src_mask):
        # 1. 멀티헤드 어텐션 
        _x = x # 입력 복사
        x = self.attention(q = x, k = x, v = x, mask = src_mask)
        
        # 2. Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. 포지션와이쥬 피드포워드
        _x =  x # 입력 미리 복사
        x = self.ffn(x)
        
        # 4. Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    

class Encoder(nn.Module):
    def __init__():