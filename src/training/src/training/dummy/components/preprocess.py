from collections.abc import Callable

from kfp import dsl
from kfp.dsl import PipelineTask


def build_preprocess_func(image: str) -> Callable[[str], PipelineTask]:
    @dsl.component(base_image=image)
    def preprocess(message: str) -> None:
        import logging
        logging.info(message)
    return preprocess
