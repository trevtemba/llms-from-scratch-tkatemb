import re
import tiktoken

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
            
        # Use regex to split by punctuation, and whitespace to get individual words/punctuation token 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Loop through every token and if it's not whitespace (.strip returns null if it's empty) then return the stripped version of the token
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else f"<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


with open(file="/home/trev/Documents/GitHub/llms-from-scratch-tkatemb/tkatemb/tokenization/the_verdict.txt", mode="r", encoding="utf-8") as f:
    raw_text = f.read() 

print(f"Total number of characters: {len(raw_text)}")
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Total number of tokens: {len(preprocessed)}")
print(preprocessed[:30])

all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
tokenizer = SimpleTokenizerV2(vocab=vocab)
text = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."

ids = tokenizer.encode(text=text)
print(ids)

words = tokenizer.decode(ids)
print(words)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)


print(tokenizer.decode(tokenizer.encode(text)))

# Byte-pair encoding:P

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)