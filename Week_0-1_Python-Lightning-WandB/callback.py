import pytorch_lightning as pl
import torch
import pandas as pd

from data import DataModule
import wandb

class SamplesVisualisationLogger(pl.Callback):
    def __init__(self, datamodule):
        super().__init__()

        self.datamodule = datamodule
        self.datamodule.prepare_data()

    def on_validation_end(self, trainer, pl_module):
        val_batch = next(iter(self.datamodule.val_dataloader()))
        # print(val_batch)
        sentences = val_batch["sentence"]

        logits = pl_module(val_batch["input_ids"], val_batch["attention_mask"])
        preds = torch.argmax(logits, 1)
        labels = val_batch["label"]

        df = pd.DataFrame(
            {"Sentence": sentences, "Label": labels.cpu().numpy(), "Predicted": preds.cpu().numpy()}
        )

        wrong_df = df[df["Label"] != df["Predicted"]]
        trainer.logger.experiment.log(
            {
                "examples": wandb.Table(dataframe=wrong_df, allow_mixed_types=True),
                "global_step": trainer.global_step,
            }
        )

if __name__ == '__main__':
    data = DataModule()
    sample_visualizer = SamplesVisualisationLogger(data)
    sample_visualizer.on_validation_end(None, None)