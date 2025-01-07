# src/model/model.py
from transformers import BartForConditionalGeneration, AutoTokenizer
from pytorch_lightning import LightningModule
from src.utils.metrics import RougeEvaluator
import torch

class DialogueSummarizer(LightningModule):
    def __init__(self, model_name: str, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.rouge_evaluator = RougeEvaluator()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=(input_ids > 0), labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        outputs = self(input_ids=input_ids, attention_mask=(input_ids > 0), labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss)

        # Decode predictions and references for ROUGE evaluation
        preds = self.model.generate(input_ids)
        decoded_preds = [self.tokenizer.decode(p, skip_special_tokens=True) for p in preds]
        decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

        rouge_scores = self.rouge_evaluator.compute_rouge(decoded_preds, decoded_labels)
        self.log("val_rouge-1", rouge_scores["rouge-1"])
        self.log("val_rouge-2", rouge_scores["rouge-2"])
        self.log("val_rouge-l", rouge_scores["rouge-l"])

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
