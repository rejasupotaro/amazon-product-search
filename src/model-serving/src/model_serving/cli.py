import hydra
from omegaconf import DictConfig

from model_serving.export.exporters import ONNXModelExporter, TorchScriptModelExporter


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def export(cfg: DictConfig) -> None:
    TorchScriptModelExporter(cfg).export()
    ONNXModelExporter(cfg).export()


if __name__ == "__main__":
    export()
