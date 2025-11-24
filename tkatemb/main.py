import tiktoken
import torch
import GPT
import torch.nn as nn
from train_eval import train_util as util
from train_eval import dataloader

GPT_CONFIG_124M = {
    "vocab_size": 50257, #vocab of 50257 words used by BPE tokenizer
    "context_length": 256, # max num of input tokens model can handle via the positional embeddings
    "emb_dim": 768,# each token will be transformed into a 768 dim vector,
    "n_heads": 12, # amount of attention heads in mha
    "n_layers": 12, # number of transformer blocks in the model
    "drop_rate": 0.1, #dropout reate
    "qkv_bias": False # whether to include a bias vector in linear layers of the mha attention for query, key and value computations
}

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    # model = DummyGPT.DummyGPTModel(GPT_CONFIG_124M)
    # print(batch)

    # torch.manual_seed(123)
    # logits = model(batch)
    # print("Output shape:", logits.shape)
    # print(logits)

    # torch.manual_seed(123)
    # batch_example = torch.randn(2, 5)
    # layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    # out = layer(batch_example)
    # print(out)

    # mean = out.mean(dim = -1, keepdim = True)
    # var = out.var(dim = -1, keepdim = True)

    # print("mean:", mean)
    # print("variance:", var)
    
    # out_norm = (out - mean) / torch.sqrt(var)
    # mean = out_norm.mean(dim=-1, keepdim=True)
    # var = out_norm.var(dim=-1, keepdim=True)
    # print("Normalized layer outputs:\n", out_norm)
    # torch.set_printoptions(sci_mode=False)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    # ln = GPT.LayerNorm(emb_dim = 5)
    # out_ln = ln(batch_example)
    # mean = out_ln.mean(dim=-1, keepdim=True)
    # var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    # torch.set_printoptions(sci_mode=False)
    # print("Mean:\n", mean)
    # print("Variance:\n", var)

    torch.manual_seed(123)
    model = GPT.GPTModel(GPT_CONFIG_124M)
    model.eval()

    # # compute parameter count
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")
    # total_mem_bytes = total_params * 4
    # total_mem_mb = total_mem_bytes / (1024 * 1024)
    # print(f"Total size of model: {total_mem_mb:.2f} MB")

    # start_context = "Every effort moves you"
    
    # token_ids = GPT.generate_text_simple(
    #     model=model,
    #     idx=util.text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M["context_length"],
    # )
    # print("Output text:\n", util.token_ids_to_text(token_ids, tokenizer))
    
    # inputs = torch.tensor([[16833, 3626, 6100], # "every effort moves",
    #                        [40, 1107, 588]]) # "I really like"
    
    # targets = torch.tensor([[3626, 6100, 345], # "effort moves you"
    #                         [1107, 588, 11311]]) # "really like chocolate"
    
    # with torch.no_grad():
    #     logits = model(inputs)

    # probas = torch.softmax(logits, dim=-1)
    # print(probas.shape)

    # token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    # print("Token IDs:\n", token_ids)

    # print(f"Targets batch 1: {util.token_ids_to_text(targets[0], tokenizer)}")
    # print(f"Outputs batch 1: {util.token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    # # Manual cross entropy
    # text_idx = 0
    # target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 1:", target_probas_1)

    # text_idx = 1
    # target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    # print("Text 2:", target_probas_2)

    # log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    # print(log_probas)
    # avg_log_probas = torch.mean(log_probas)
    # print(avg_log_probas)

    # neg_avg_log_probas = avg_log_probas * -1
    # print(neg_avg_log_probas)

    # # Industry
    # logits_flat = logits.flatten(0, 1)
    # targets_flat = targets.flatten()
    # print("Flattened logits:", logits_flat.shape)
    # print("Flattened targets:", targets_flat.shape)

    # loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    # print(loss)

    file_path = "tkatemb/tokenization/the_verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = dataloader.create_dataloader_v1(
        txt=train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    
    
    val_loader = dataloader.create_dataloader_v1(
        txt=val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    start_context = "Every effort moves you"

    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
        
    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = util.train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context=start_context, tokenizer=tokenizer
    )

    model.to("cpu")
    model.eval()
    torch.manual_seed(123)
    token_ids = util.generate(model=model, idx=util.text_to_token_ids("Every effort moves you", tokenizer),
                              max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"], top_k= 25, temperature=1.4)
    print("Output text:\n", util.token_ids_to_text(token_ids, tokenizer))


    


