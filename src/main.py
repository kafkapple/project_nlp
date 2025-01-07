import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import hydra
import warnings
import torchvision
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from src.logger.logger import get_logger
from src.data.datamodule import DialogueDataModule
from src.data.download import download_dialogsum
from src.data.preprocess import preprocess_dialogsum
from src.model.model import DialogueSummarizer
from src.trainer.trainer import get_trainer

# 모든 torchvision 관련 경고 비활성화
warnings.filterwarnings(action = "ignore", category=UserWarning, module="torchvision")
# Beta transforms 경고 비활성화
torchvision.disable_beta_transforms_warning()

@hydra.main(version_base="1.2", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger("DialogueSummarization")
    logger.info("Starting Dialogue Summarization...")

    # Ensure dataset is available
    data_dir = os.path.dirname(cfg.data.train_file)
    os.makedirs(data_dir, exist_ok=True)
    download_dialogsum(data_dir)
    preprocess_dialogsum(data_dir)

    # Logger
    wandb_logger = WandbLogger(project=cfg.wandb.project)

    # DataModule
    data_module = DialogueDataModule(
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        tokenizer_name=cfg.model.tokenizer,
        batch_size=cfg.data.batch_size,
        max_len=cfg.data.max_len,
    )

    # Model
    model = DialogueSummarizer(
        model_name=cfg.model.name,
        learning_rate=cfg.model.learning_rate
    )

    # Trainer
    trainer = get_trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger
    )

    # Train
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
