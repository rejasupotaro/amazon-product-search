from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class MetricLogger(Callback):
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics["train_loss"]
        self.metrics.append(
            {
                "epoch": epoch,
                "metric_name": "train_loss",
                "value": round(float(loss.detach().cpu().numpy()), 4),
            }
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        loss = trainer.callback_metrics["val_loss"]
        self.metrics.append(
            {
                "epoch": epoch,
                "metric_name": "val_loss",
                "value": round(float(loss.detach().cpu().numpy()), 4),
            }
        )
