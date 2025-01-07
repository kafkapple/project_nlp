# src/trainer/trainer.py
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

def get_trainer(max_epochs, logger):
    return Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=logger
    )
