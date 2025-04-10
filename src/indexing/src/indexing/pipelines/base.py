import logging
import os
from abc import ABC, abstractmethod

import apache_beam as beam

from indexing.options import IndexerOptions


class BasePipeline(ABC):
    def __init__(self, options: IndexerOptions) -> None:
        self.options = options
        logging.getLogger().setLevel(logging.INFO)
        # Disable the warning as suggested:
        # ```
        # huggingface/tokenizers: The current process just got forked,
        # after parallelism has already been used. Disabling parallelism to avoid deadlocks...
        # To disable this warning, you can either:
        # - Avoid using `tokenizers` before the fork if possible
        # - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        # ```
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    @abstractmethod
    def build(self, options: IndexerOptions) -> beam.Pipeline:
        raise NotImplementedError

    def run(self) -> None:
        pipeline = self.build(self.options)
        result = pipeline.run()
        result.wait_until_finish()
