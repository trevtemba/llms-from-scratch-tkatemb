import tiktoken
import torch
import DummyGPT
import torch.nn as nn
GPT_CONFIG_124M = {
    "vocab_size": 50257, #vocab of 50257 words used by BPE tokenizer
    "context_length": 1024, # max num of input tokens model can handle via the positional embeddings
    "emb_dim": 768,# each token will be transformed into a 768 dim vector,
    "n_heads": 12, # amount of attention heads in mha
    "n_layers": 12, # number of transformer blocks in the model
    "drop_rate": 0.1, #dropout reate
    "qkv_bias": False # whether to include a bias vector in linear layers of the mha attention for query, key and value computations
}

if __name__ == "__main__":
    # tokenizer = tiktoken.get_encoding("gpt2")
    # batch = []
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"

    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)

    # model = DummyGPT.DummyGPTModel(GPT_CONFIG_124M)
    # print(batch)

    # torch.manual_seed(123)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim = -1, keepdim = True)
    var = out.var(dim = -1, keepdim = True)

    print("mean:", mean)
    print("variance:", var)
    
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)