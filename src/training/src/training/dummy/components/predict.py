from kfp import dsl


def build_predict_func() -> dsl.ContainerOp:
    @dsl.container
    def predict() -> None:
        print("Predict")

    return predict
