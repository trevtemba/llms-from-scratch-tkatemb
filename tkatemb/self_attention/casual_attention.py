import torch
import torch.nn as nn
import self_attention as sa

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]]   # step    (x^6)
)

torch.manual_seed(123)

d_in = inputs.shape[1]
d_out = 2
sa_v2 = sa.SelfAttention_V2(d_in=d_in, d_out=d_out, qkv_bias=False)

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

context_length = attn_scores.shape[0]
# # Naive
# attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
# print(attn_weights)

# mask = torch.tril(torch.ones(context_length, context_length))

# masked_simple = attn_weights * mask
# row_sums = masked_simple.sum(dim=-1, keepdim=True)
# masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

# Efficient

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill_(mask.bool(), -torch.inf)
masked_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(masked_weights)

# Applying dropout

dropout = torch.nn.Dropout(0.5)
print(dropout(masked_weights))