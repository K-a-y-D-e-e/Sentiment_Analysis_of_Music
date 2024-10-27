import pandas as pd
from transformers import BertTokenizer

# Load your dataset (replace with your file path)
data_path = "path_to_your_dataset.csv"
df = pd.read_csv(data_path)

# Inspect dataset
print(df.head())

# Example columns in your dataset: 'lyrics' and 'emotion'
lyrics_texts = df['lyrics'].tolist()
labels = df['emotion'].tolist()

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare the DataLoader with Your Dataset

from torch.utils.data import DataLoader

# Define max_length based on the average length of your lyrics (128 is a common choice)
max_length = 128

# Create dataset and dataloader
dataset = LyricsDataset(lyrics_texts, labels, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
