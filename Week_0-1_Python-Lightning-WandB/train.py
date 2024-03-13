from model import ColaModel
from data import DataModule
from callback import SamplesVisualisationLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="./configs", config_name='config')
def main(cfg):
    # print(cfg.model.name)
    wandb_logger = WandbLogger(project="MLOps Basics")

    cola_data = DataModule(
        model_name=cfg.model.name, batch_size=cfg.processing.batch_size
    )
    cola_model = ColaModel(
        model_name=cfg.model.name, lr=cfg.training.lr
    )

    visualize_callback = SamplesVisualisationLogger(cola_data)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(default_root_dir="logs",
                         max_epochs=cfg.training.max_epochs,
                         accelerator="auto",
                         fast_dev_run=False,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, visualize_callback, early_stopping_callback],
                         )

    trainer.fit(cola_model, cola_data)

if __name__ == "__main__":
    main()