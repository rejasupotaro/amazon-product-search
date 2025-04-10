import os
from typing import Any, Dict, Iterator

import apache_beam as beam
import pandas as pd
from data_source import Locale


class AddImageUrlFn(beam.DoFn):
    """This is a Beam DoFn that adds image URL to product.

    The data source is extracted from https://github.com/shuttie/esci-s?tab=readme-ov-file

    Args:
        filepath (str): Path to the parquet file that contains `asin`, `locale`, and `image`.
        locale (Locale): Locale of the products.

    Attributes:
        asin_to_image_url (Dict[str, str]): A dictionary that maps ASIN to image URL.
    """

    def __init__(self, filepath: str, locale: Locale) -> None:
        super().__init__()
        self.asin_to_image_url = self.load_image_url_dict(filepath, locale)

    @staticmethod
    def load_image_url_dict(filepath: str, locale: Locale) -> Dict[str, str]:
        if not os.path.exists(filepath):
            # Return an empty dictionary if the file does not exist.
            return {}

        df = pd.read_parquet(filepath)
        df = df[df["locale"] == locale]
        return dict(zip(df["asin"], df["image"], strict=True))

    def process(self, product: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if "image_url" in product:
            yield product
            return

        if product["product_id"] in self.asin_to_image_url:
            product["image_url"] = self.asin_to_image_url[product["product_id"]]
        yield product
