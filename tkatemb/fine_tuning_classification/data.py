import pandas as pd
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import tiktoken
from dataset import SpamDataset

# Configuration
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

df = pd.read_csv(data_file_path, sep="\t", header=None, names = ["Label", "Text"])

def random_split(df, train_frac, validation_frac):
    """
    Randomly split a DataFrame into training, validation, and test sets.
    
    Args:
        df: DataFrame to split
        train_frac: Fraction of data for training (e.g., 0.7 for 70%)
        validation_frac: Fraction of data for validation (e.g., 0.1 for 10%)
        
    Returns:
        tuple: (train_df, validation_df, test_df)
        
    Note:
        Test fraction is implicitly 1 - train_frac - validation_frac
    """
    # Shuffle the entire DataFrame and reset index
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    # Calculate the split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    # Split the DataFrame into three sets
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    return train_df, validation_df, test_df
    
def create_balanced_dataset(df):
    """
    Create a balanced dataset by sampling ham messages to match spam count.
    
    Args:
        df: DataFrame with 'Label' column containing 'spam' and 'ham' labels
        
    Returns:
        DataFrame with equal numbers of spam and ham messages
    """
    # Get the number of spam messages
    num_spam = df[df["Label"] == "spam"].shape[0]  # 1
    
    # Sample ham messages to match spam count
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )  # 2
    
    # Combine ham subset with all spam messages
    balanced_df = pd.concat([
        ham_subset,
        df[df["Label"] == "spam"]
    ])  # 3
    
    return balanced_df

# Create balanced dataset and display label distribution
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

create_balanced_dataset(df)

balanced_df["Label"] = balanced_df["Label"].map({"ham":0, "spam": 1})


# Split the balanced dataset: 70% train, 10% validation, 20% test
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Display the sizes of each split
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(validation_df)} samples")
print(f"Test set: {len(test_df)} samples")

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
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