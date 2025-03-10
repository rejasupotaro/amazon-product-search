from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class MetricLoggerPL(Callback):
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


class MetricLoggerST:
    def __init__(self, metric_name: str) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.metric_name = metric_name

    def __call__(self, score: float, epoch: int, steps: int) -> None:
        self.metrics.append(
            {
                "epoch": epoch,
                "metric_name": self.metric_name,
                "value": score,
            }
        )
