import torch
from torch.utils.data import Dataset
import pandas as pd

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        
        # Pretokenizes text
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # Truncates text if there is a max length set, or finds longest encode length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        
        #Pads sequences to the longest sequence (appends pad tokens to the given encoded text)
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


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
