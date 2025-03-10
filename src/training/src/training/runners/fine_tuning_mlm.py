from typing import Any, Optional, Union

import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from training.data_modules.product_mlm_data_module import ProductMLMDataModule
from training.modules.mlm_fine_tuner import MLMFineTuner
from training.shared.metric_logger import MetricLoggerPL


def run(
    project_dir: str,
    input_filename: str,
    bert_model_name: str,
    learning_rate: float = 1e-4,
    mlm_probability: float = 0.1,
    batch_size: int = 32,
    num_sentences: Optional[int] = None,
    max_epochs: int = 1,
    devices: Union[list[int], str, int] = "auto",
) -> list[dict[str, Any]]:
    data_dir = f"{project_dir}/data"
    models_dir = f"{project_dir}/models"

    df = pd.read_parquet(f"{data_dir}/{input_filename}")

    fine_tuner = MLMFineTuner(bert_model_name, learning_rate)
    data_module = ProductMLMDataModule(bert_model_name, df, mlm_probability, batch_size, num_sentences)
    metric_logger = MetricLoggerPL()
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{models_dir}/checkpoints/{bert_model_name}",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        precision="16-mixed",
        devices=devices,
        callbacks=[metric_logger, model_checkpoint],
    )
    trainer.fit(fine_tuner, data_module)

    output_dir = f"{models_dir}/fine_tuned/{bert_model_name}"
    model = fine_tuner.model
    tokenizer = data_module.tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return metric_logger.metrics
