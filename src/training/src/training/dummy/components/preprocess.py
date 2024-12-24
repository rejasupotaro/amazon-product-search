from kfp import dsl


def build_preprocess_func(image: str) -> dsl.ContainerOp:
    @dsl.container(
        name="preprocess",
        image=image,
    )
    def preprocess(message: str) -> None:
        import logging

        logging.info(message)

    return preprocess(message="Preprocess")
