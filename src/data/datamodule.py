from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from src.data.dataset import DialogueDataset

class DialogueDataModule(LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        tokenizer_name: str,
        batch_size: int,
        max_len: int,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DialogueDataset(
                self.train_file,
                self.tokenizer,
                self.max_len
            )
            self.val_dataset = DialogueDataset(
                self.val_file,
                self.tokenizer,
                self.max_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )
