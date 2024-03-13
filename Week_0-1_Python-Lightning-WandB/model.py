import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-3):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_class = 2

        self.validation_step_output = []

        self.train_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.val_accuracy_metric = torchmetrics.Accuracy(task="binary")
        self.f1_metric = torchmetrics.F1Score(num_classes=self.num_class, task="binary")
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_class, task="binary"
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_class, task="binary"
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro", task="binary")
        self.recall_micro_metric = torchmetrics.Recall(average="micro", task="binary")

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])

        preds = torch.argmax(logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch['label'])

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["label"])
        _, preds = torch.max(logits, dim=1)

        # val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        # val_acc = torch.tensor(val_acc)
        valid_acc = self.val_accuracy_metric(preds, batch["label"])
        precision_macro = self.precision_macro_metric(preds, batch["label"])
        recall_macro = self.recall_macro_metric(preds, batch["label"])
        precision_micro = self.precision_micro_metric(preds, batch["label"])
        recall_micro = self.recall_micro_metric(preds, batch["label"])
        f1 = self.f1_metric(preds, batch["label"])

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        self.log("valid/f1", f1, prog_bar=True)

        self.validation_step_output.append({"labels": batch["label"], "logits": logits})

    def on_validation_epoch_end(self):
        labels = torch.cat([x["labels"] for x in self.validation_step_output])
        logits = torch.cat([x["logits"] for x in self.validation_step_output])

        preds = torch.argmax(logits, 1)

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

        # cm = confusion_matrix(labels.numpy(), preds.numpy())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

if __name__ == "__main__":
    from data import DataModule

    model = ColaModel()
    data = DataModule()

    trainer = pl.Trainer(max_epochs=0,
                         accelerator="auto",
                         fast_dev_run=True,
                         )

    trainer.fit(model, data)