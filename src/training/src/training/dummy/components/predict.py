from collections.abc import Callable

from kfp import dsl
from kfp.dsl import PipelineTask


def build_predict_func(image: str) -> Callable[[], PipelineTask]:
    @dsl.component(base_image=image)
    def predict() -> None:
        print("Predict")
    return predict
