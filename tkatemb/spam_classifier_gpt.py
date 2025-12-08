import pandas as pd
import time
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import tiktoken
from fine_tuning_classification import dataset
from gpt_download import download_and_load_gpt2
from load_weights import load_weights_util
from GPT import GPTModel
from train_eval import train_util as util
from fine_tuning_classification import classification_util

# Configuration
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

df = pd.read_csv(data_file_path, sep="\t", header=None, names = ["Label", "Text"])

# Create balanced dataset and display label distribution
balanced_df = dataset.create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

dataset.create_balanced_dataset(df)

balanced_df["Label"] = balanced_df["Label"].map({"ham":0, "spam": 1})


# Split the balanced dataset: 70% train, 10% validation, 20% test
train_df, validation_df, test_df = dataset.random_split(balanced_df, 0.7, 0.1)

# Display the sizes of each split
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(validation_df)} samples")
print(f"Test set: {len(test_df)} samples")

train_df, validation_df, test_df = dataset.random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = dataset.SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = dataset.SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = dataset.SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

print(train_dataset.max_length)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

for input_batch, target_batch in train_loader:
    pass

print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape)

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
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

text_1 = "Every effort moves you"
token_ids = util.generate_text_simple(
    model=model,
    idx=util.text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(util.token_ids_to_text(token_ids, tokenizer))

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)

#Makes final layer norm/last transformer block trainable
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)

print("Outputs:", outputs)
print("Outputs dimensions", outputs.shape)

print("Last output token:", outputs[:,-1,:])

probas = torch.softmax(outputs[:,-1,:], dim = -1)
label = torch.argmax(probas)
print("Class label:", label.item())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Determine teh classification accuracy across various datasets estimated form 10 batches for efficiency
torch.manual_seed(123)
train_accuracy = classification_util.calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)

val_accuracy = classification_util.calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)
test_accuracy = classification_util.calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# similar to calculating teh trainin accuracy we can no copmute the intial loss for each data set.
with torch.no_grad():
    train_loss = classification_util.calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    
    val_loss = classification_util.calc_loss_loader(
        val_loader, model, device, num_batches=5
    )
    test_loss = classification_util.calc_loss_loader(
        test_loader, model, device, num_batches=5
    )

print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    classification_util.train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time)/60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

train_accuracy = classification_util.calc_accuracy_loader(train_loader, model, device)
val_accuracy = classification_util.calc_accuracy_loader(val_loader, model, device)
test_accuracy = classification_util.calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Val accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Using fine tuned classifier.``
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(
    classification_util.classify_review(
        text_1,
        model,
        tokenizer,
        device,
        max_length=train_dataset.max_length
    )
)

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(
    classification_util.classify_review(
        text_2,
        model,
        tokenizer,
        device,
        max_length=train_dataset.max_length
    )
)

torch.save(model.state_dict(), "review_classifier.pth")

# If you want to load it do:
# model_state_dict = torch.load(
#     "review_classifier.pth",
#     map_location=device
# )
# model.load_state_dict(model_state_dict)