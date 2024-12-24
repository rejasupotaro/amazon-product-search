from abc import ABC, abstractmethod

from kfp.dsl.container_component_class import ContainerComponent


class BaseComponent(ABC):
    def __init__(self, image: str) -> None:
        self.image = image
        self.component_func: ContainerComponent = self.build_component_func(image=image)

    @abstractmethod
    def build_component_func(self, image: str) -> ContainerComponent:
        raise NotImplementedError

