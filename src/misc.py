import pandas as pd

DATA_DIR = "./data"


def split_dataset_by_locale():
    filename = "product_catalogue-v0.3"
    filepath = f"{DATA_DIR}/{filename}.csv.zip"
    print(f"Load catalog from {filepath}...")
    df = pd.read_csv(filepath)

    locales = df["product_locale"].unique()
    print(f"The dataset contains locales: {locales}")

    for locale in locales:
        filtered_df = df[df["product_locale"] == locale]
        filepath = f"{DATA_DIR}/{filename}_{locale}.csv.zip"
        filtered_df.to_csv(filepath)
        print(f"A catalog dataset (locale: {locale}) containing {len(filtered_df)} rows is saved to {filepath}")
