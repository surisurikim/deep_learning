# https://youtu.be/QCJQG4DuHT0?si=NLsvZvFGyoYs2fIJ

#%%
import numpy as np
import math

L, d_k, d_v = 4, 8, 8
# L : 4글자 문장이라고 생각해 봅시다 (ex. My name is Suri)
# k, v는 임의로 8이라고 생각해 봅시다
# shape (4, 8) 랜덤 텐서 q, k, v 3개 생성

q = np.random.randn(L, d_k)
# print("Q\n", q)
# print(q.shape)

k = np.random.randn(L, d_k)
# print("K\n", k)
# print(k.shape)

v = np.random.randn(L, d_v)
# print("V\n", v)
# print(v.shape)

# 1. matmul 계산
qkt = np.matmul(q, k.T)
print("Q * K(transposed)\n", qkt)

# 2. Scaling
scaling = math.sqrt(d_k)
scaled = qkt / scaling
print("\nScaled\n", scaled)
print("\n", q.var(), k.var(), scaled.var())
print("\n")

# 3. Masking (only in Decoder)
mask = np.tril(np.ones((L, L))) # 오른쪽+상단은 0, 왼쪽+하단+대각선은 1
print(mask)
print("\n")

mask[mask == 0] = -np.infty
mask[mask == 1] = 0 # 0인 곳은 -inf(마이너스 무한대), 1인 곳은 0으로 바꿈
print(mask)
print("\n")

scaled = scaled + mask
print("-------------- masked --------------")
print(scaled) # masked 완료! 
print("\n")

# 4. Softmax
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis = -1)).T

print("-------------- attention --------------")
attention = softmax(scaled)
print(attention)
print("\n")


# 5. Matmul with V
new_v = np.matmul(attention, v)
print("-------------- QKT + matmul with V --------------")
print(new_v)
print('before')
print(v)


########################## final product (전체 프로세스 함수화)

def scaled_dot_product_attention(q, k, v, mask = None):
    d_k = q.shape[-1] # 스케일링용
    scaled = np.matmul(q, k.T) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled + mask
    attention = softmax(scaled)
    out = np.matmul(attention, v)
    return out, attention

values, attention = scaled_dot_product_attention(q, k, v, mask = None)
print("------------- final -------------")
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("New V\n", values)
print("Attention\n", attention)



# %%
