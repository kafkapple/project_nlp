# src/data/dataset.py
import json
import pandas as pd
from torch.utils.data import Dataset

class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dialogue = item['dialogue']
        summary = item['summary']

        # 입력 텍스트와 타겟 텍스트를 토크나이징
        inputs = self.tokenizer(
            dialogue,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }
