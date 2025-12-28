import json
import os
import urllib.request
import torch
import tiktoken
from torch.utils.data import DataLoader
from functools import partial
from fine_tuning_instructions import util
from fine_tuning_instructions import dataset

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