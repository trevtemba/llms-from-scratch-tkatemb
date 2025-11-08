import torch
# inputs = torch.tensor(
#     [[0.43, 0.15, 0.89],  # Your    (x^1)
#     [0.55, 0.87, 0.66],  # journey (x^2)
#     [0.57, 0.85, 0.64],  # starts  (x^3)
#     [0.22, 0.58, 0.33],  # with    (x^4)
#     [0.77, 0.25, 0.10],  # one     (x^5)
#     [0.05, 0.80, 0.55]]   # step    (x^6)
# )

# query = inputs[1]
# attn_scores_2 = torch.empty(inputs.shape[0])
# for i, x_i in enumerate(inputs):
#     attn_scores_2[i] = torch.dot(x_i, query)
# print(attn_scores_2)

# # normalization simple
# # attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# # print("Attention weights:", attn_weights_2_tmp)
# # print("Sum:", attn_weights_2_tmp.sum())

# # industry standard
# attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

# context_vec_2 = torch.zeros(query.shape)
# for i, x_i in enumerate(inputs):
#     context_vec_2 += x_i * attn_weights_2[i]
# print(context_vec_2)

# Naive way
# attn_scores = torch.empty(6,6)
# for i, x_i in enumerate(inputs):
#     for j, x_j in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(x_i, x_j)
# print(attn_scores)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your    (x^1)
    [0.55, 0.87, 0.66],  # journey (x^2)
    [0.57, 0.85, 0.64],  # starts  (x^3)
    [0.22, 0.58, 0.33],  # with    (x^4)
    [0.77, 0.25, 0.10],  # one     (x^5)
    [0.05, 0.80, 0.55]]   # step    (x^6)
)

# Better way (matrix multiplication)
attn_scores = inputs @ inputs.T
print("attention scores:\n", attn_scores)

# We use dim=-1 because we want the softmax to be applied to the LAST dimension of the tensor
# (2d tensor, it normalizes across the columns so that each value in a ROW sum up to 1)
attn_weights = torch.softmax(attn_scores, dim=-1)
print("attention weights:\n", attn_weights)

context_vectors = attn_weights @ inputs
print("context vectors:\n", context_vectors)



