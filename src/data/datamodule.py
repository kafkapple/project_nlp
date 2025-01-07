import json
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.dataset import DialogueDataset

class DialogueDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        tokenizer_name: str,
        batch_size: int = 8,
        max_len: int = 512,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage=None):
        # 데이터 로드
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(self.val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)

        # Dataset 생성
        self.train_dataset = DialogueDataset(
            train_data, self.tokenizer, self.max_len
        )
        self.val_dataset = DialogueDataset(
            val_data, self.tokenizer, self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
