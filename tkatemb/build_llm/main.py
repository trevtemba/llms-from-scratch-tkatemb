GPT_CONFIG = {
    "vocab_size": 50257, #vocab of 50257 words used by BPE tokenizer
    "context_length": 1024, # max num of input tokens model can handle via the positional embeddings
    "emb_dim": 768,# each token will be transformed into a 768 dim vector,
    "n_heads": 12, # amount of attention heads in mha
    "n_layers": 12, # number of transformer blocks in the model
    "drop_rate": 0.1, #dropout reate
    "qkv_bias": False # whether to include a bias vector in linear layers of the mha attention for query, key and value computations
}