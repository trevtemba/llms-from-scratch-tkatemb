import json
import time
import os
import urllib.request
import torch
import re
import tiktoken
from torch.utils.data import DataLoader
from functools import partial
from fine_tuning_instructions import util
from fine_tuning_instructions import dataset
from gpt_download import download_and_load_gpt2
from load_weights import load_weights_util
from train_eval import train_util as train_util
from GPT import GPTModel
from tqdm import tqdm
def download_and_load_file(file_path, url):
    """Download a JSON file from a URL if it doesn't exist locally, then load it."""
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    return data


if __name__ == "__main__":
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    
    data = download_and_load_file(file_path, url)
    print(f"Number of entries: {len(data)}")
    
    model_input = util.format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[ 50]['output']}"
    print(model_input + desired_response)

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Train set length: ", len(train_data))
    print("Test set length: ", len(test_data))
    print("Val set length: ", len(val_data))

    tokenizer = tiktoken.get_encoding("gpt2")

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (
        inputs_1,
        inputs_2,
        inputs_3
    )

    inputs, targets = util.custom_collate_fn(batch)
    print(inputs)
    print(targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    customized_collate_fn = partial(
        util.custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = dataset.InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = dataset.InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    test_dataset = dataset.InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    # Loading pretrained GPT model 355M
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG = {
        "vocab_size":50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {
            "emb_dim": 768,
            "n_layers": 12,
            "n_heads": 12
        },
        "gpt2-medium (355M)": {
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16
        },
        "gpt2-large (774M)": {
            "emb_dim": 1280,
            "n_layers": 36,
            "n_heads": 20
        },
        "gpt2-xl (1558M)": {
            "emb_dim": 1600,
            "n_layers": 48,
            "n_heads": 25
        },
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # Get the last word and remove parentheses
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_util.load_weights_into_gpt(model, params)
    model.eval()

    torch.manual_seed(123)
    input_text = util.format_input(val_data[0])
    print(input_text)

    token_ids = train_util.generate(
        model=model,
        idx=train_util.text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )

    generated_text = train_util.token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

    model.to(device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = train_util.calc_loss_loader(
            train_loader, model, device, num_batches=5
        )
        val_loss = train_util.calc_loss_loader(
            val_loader, model, device, num_batches=5
        )
    
    print("Training loss:", train_loss)
    print("Val loss:", val_loss)
    
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_util.train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=util.format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training copmleted in {execution_time_minutes:.2f} minutes")

    # Print the model responses alongside the expected test set answers
    # For the first three test set entries

    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = util.format_input(entry)
        token_ids = train_util.generate(
            model=model,
            idx=train_util.text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = train_util.token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\Model response:\n>> {response_text.strip()}")
        print("--------------------------")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = util.format_input(entry)

        token_ids = train_util.generate(
            model=model,
            idx=train_util.text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = train_util.token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

        with open("instruction-data-with-response.json", "w") as file:
            json.dump(test_data, file, indent=4)

        print(test_data[0])

        file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Model saved as {file_name}")