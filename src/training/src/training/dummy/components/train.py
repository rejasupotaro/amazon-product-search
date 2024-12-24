
from kfp import dsl
from kfp.dsl.container_component_class import ContainerComponent


def build_train_func(image: str) -> ContainerComponent:
    @dsl.container_component
    def train() -> dsl.ContainerSpec:
        return dsl.ContainerSpec(
            image=image,
            command=["python"],
            args=["-c", "print('Hello World')"],
        )
    return train
