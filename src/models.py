from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from src.layers import Parameter, Layer, LinearLayer


class Model(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def params(self) -> List[Parameter]:
        params_list = []
        for layer in self.__dict__.values():
            if isinstance(layer, Layer):
                params_list.extend(layer.params())
        return params_list
