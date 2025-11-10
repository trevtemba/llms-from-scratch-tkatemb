import torch
import self_attention as sa

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]]   # step    (x^6)
)

d_in = inputs.shape[1] # input embedding size
d_out = 2 # output embedding size these are usually the same, but for demo purposes they are different here.

# x_2 = inputs[1] # query of 2nd element
# torch.manual_seed(123)
# W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
# W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# query_2 = x_2 @ W_query
# key_2 = x_2 @ W_key
# value_2 = x_2 @ W_value
# print(query_2)

# keys = inputs @ W_key
# values = inputs @ W_value
# print("keys.shape: ", keys.shape)
# print("values.shape: ", values.shape)

# # keys_2 = keys[1]
# # attn_score_22 = query_2.dot(keys_2)
# # print(attn_score_22)

# attn_score_2 = query_2 @ keys.T
# print(attn_score_2)

# d_k = keys.shape[-1]
# attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)

# context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

# torch.manual_seed(123)
# sa_v2 = sa.SelfAttention_V2(d_in=d_in, d_out=d_out, qkv_bias=False)
# print(sa_v2(inputs))

batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)
torch.manual_seed(123)
context_length = batch.shape[1]
d_in = batch.shape[-1]
d_out = 2

# ca = sa.CasualAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0)
# context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)

mha = sa.MultiHeadAttentionWrapper(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)