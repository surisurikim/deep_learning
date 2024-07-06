"""
    @ author : Seul Kim
    @ when : Jan 06, 2024
    @ contact : niceonesuri@gmail.com
    @ blog : https://smartest-suri.tistory.com/
"""


from layers import TokenEmbedding, PositionalEncoding, TransformerEmbedding, PositionwiseFeedForward, LayerNorm
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
        x = self.norm1(x + _x) # norm(add)
        
        # 3. positionwise feed forward!
        _x =  x # 입력 미리 복사
        x = self.ffn(x)
        
        # 4. Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, 
                 ffn_hidden, n_heads, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model = d_model,
                                        max_len = max_len,
                                        vocab_size = enc_voc_size,
                                        drop_prob = drop_prob,
                                        device = device)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                  n_heads = n_heads,
                                                  drop_prob = drop_prob)
                                        for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x