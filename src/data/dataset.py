# src/data/dataset.py
import json
import pandas as pd
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        """
        DialogSum Dataset for dialogue summarization.
        Args:
            file_path (str): Path to the dataset file (JSON).
            tokenizer: Pretrained tokenizer for text processing.
            max_len (int): Maximum token length.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Tokenize input dialogue and summary.
        """
        item = self.data[idx]
        dialogue = item['dialogue']
        summary = item['summary']
        
        dialogue_encoding = self.tokenizer(
            dialogue,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return dialogue_encoding['input_ids'].squeeze(), summary_encoding['input_ids'].squeeze()
