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
        
        # 최고 성능 추적을 위한 변수들
        self.best_rouge1 = 0.0
        self.best_rouge2 = 0.0
        self.best_rougel = 0.0

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # WandB에 학습 메트릭 로깅 추가
        self.log('train_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # 기본 loss 로깅
        self.log('val_loss', outputs.loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # ROUGE 평가를 위한 생성 및 디코딩
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        
        # 생성된 텍스트와 레이블 디코딩
        predictions = [self.tokenizer.decode(g, skip_special_tokens=True) 
                      for g in generated_ids]
        references = [self.tokenizer.decode(l, skip_special_tokens=True) 
                     for l in batch["labels"]]
        
        # ROUGE 점수 계산 및 텐서 변환
        rouge_scores = self.rouge_evaluator.compute_rouge(predictions, references)
        rouge_scores = {
            'rouge-1': torch.tensor(rouge_scores['rouge-1'], device=self.device),
            'rouge-2': torch.tensor(rouge_scores['rouge-2'], device=self.device),
            'rouge-l': torch.tensor(rouge_scores['rouge-l'], device=self.device)
        }
        
        # 각 배치의 ROUGE 점수 로깅
        self.log('val_rouge1', rouge_scores['rouge-1'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_rouge2', rouge_scores['rouge-2'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_rougeL', rouge_scores['rouge-l'], on_step=True, on_epoch=True, prog_bar=True)
        
        return {'val_loss': outputs.loss, 'rouge_scores': rouge_scores}

    def validation_epoch_end(self, outputs):
        # 에포크 평균 ROUGE 점수 계산
        avg_rouge1 = torch.stack([x['rouge_scores']['rouge-1'] for x in outputs]).mean()
        avg_rouge2 = torch.stack([x['rouge_scores']['rouge-2'] for x in outputs]).mean()
        avg_rougel = torch.stack([x['rouge_scores']['rouge-l'] for x in outputs]).mean()
        
        # 최고 성능 업데이트 및 로깅
        if avg_rouge1 > self.best_rouge1:
            self.best_rouge1 = avg_rouge1
            self.log('best_rouge1', self.best_rouge1)
        
        if avg_rouge2 > self.best_rouge2:
            self.best_rouge2 = avg_rouge2
            self.log('best_rouge2', self.best_rouge2)
            
        if avg_rougel > self.best_rougel:
            self.best_rougel = avg_rougel
            self.log('best_rougeL', self.best_rougel)
        
        # 현재 에포크의 평균 점수 로깅
        self.log('epoch_rouge1', avg_rouge1)
        self.log('epoch_rouge2', avg_rouge2)
        self.log('epoch_rougeL', avg_rougel)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
