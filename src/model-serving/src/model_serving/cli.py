import hydra
from omegaconf import DictConfig

from model_serving.onnx import exporter


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def export(cfg: DictConfig) -> None:
    exporter.export(cfg)


if __name__ == "__main__":
    export()
