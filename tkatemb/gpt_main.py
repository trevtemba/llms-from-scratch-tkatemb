import tiktoken
import torch
import GPT
from train_eval import train_util as util
from load_weights import load_weights_util
from gpt_download import download_and_load_gpt2
from train_eval import dataloader

settings, params = download_and_load_gpt2(
    model_size="1558M", models_dir="gpt2"
)

GPT_CONFIG_124M = {
    "vocab_size": 50257, #vocab of 50257 words used by BPE tokenizer
    "context_length": 256, # max num of input tokens model can handle via the positional embeddings
    "emb_dim": 768,# each token will be transformed into a 768 dim vector,
    "n_heads": 12, # amount of attention heads in mha
    "n_layers": 12, # number of transformer blocks in the model
    "drop_rate": 0.1, #dropout reate
    "qkv_bias": False # whether to include a bias vector in linear layers of the mha attention for query, key and value computations
}

print("Settings:", settings)
print("Params:", params.keys())

model_configs = {
    "gpt2-small(124M)": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    },
    "gpt2-medium(355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    },
    "gpt2-large(774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20
    },
    "gpt2-xl(1558M)": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25
    },
}
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2-xl(1558M)"

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPT.GPTModel(NEW_CONFIG)
gpt.train()

load_weights_util.load_weights_into_gpt(gpt, params)
gpt.to(device)



torch.manual_seed(123)
token_ids= util.generate(
    model=gpt,
    idx=util.text_to_token_ids("Clemson university is most known for", tokenizer).to(device),
    max_new_tokens=100,
    context_size=NEW_CONFIG["context_length"],
    top_k=25,
    temperature=0.25
)

print("Output text:\n", util.token_ids_to_text(token_ids, tokenizer))

# file_path = "tkatemb/tokenization/the_verdict.txt"
# with open(file_path, "r", encoding="utf-8") as file:
#     text_data = file.read()
# total_characters = len(text_data)
# total_tokens = len(tokenizer.encode(text_data))
# print("Characters:", total_characters)
# print("Tokens:", total_tokens)

# train_ratio = 0.8
# split_idx = int(train_ratio * len(text_data))
# train_data = text_data[:split_idx]
# val_data = text_data[split_idx:]

# train_loader = dataloader.create_dataloader_v1(
#     txt=train_data,
#     batch_size=2,
#     max_length=NEW_CONFIG["context_length"],
#     stride=NEW_CONFIG["context_length"],
#     drop_last=True,
#     shuffle=True,
#     num_workers=0,
# )


# val_loader = dataloader.create_dataloader_v1(
#     txt=val_data,
#     batch_size=2,
#     max_length=NEW_CONFIG["context_length"],
#     stride=128,
#     drop_last=False,
#     shuffle=False,
#     num_workers=0,
# )

# start_context = "Every effort moves you"

# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
    
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# gpt.to(device)
# optimizer = torch.optim.AdamW(
#     gpt.parameters(),
#     lr=0.0001, weight_decay=0.1
# )
# num_epochs = 10
# train_losses, val_losses, tokens_seen = util.train_model_simple(
#     gpt, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context=start_context, tokenizer=tokenizer
# )