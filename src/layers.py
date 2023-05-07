from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class Parameter:
    SCALE = 1

    def __init__(self, *args: int) -> None:
        self.weights = (np.random.rand(*args) - 0.5) * 2 * self.SCALE
        self.shape = self.weights.shape

    def __str__(self) -> str:
        return self.weights.__str__()
    
    def __repr__(self) -> str:
        return self.weights.__repr__()


class Layer(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def params(self) -> List[Parameter]:
        params_list = []
        for attribute in self.__dict__.values():
            if isinstance(attribute, Parameter):
                params_list.append(attribute)
        return params_list


class LinearLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.w = Parameter(input_size, output_size)
        self.b = Parameter(1, output_size)
    
    def forward(self, input: np.array) -> np.array:
        if len(input.shape) != 2 or input.shape[1] != self.w.shape[0]:
            raise ValueError('Input array shape in incorrect')
        
        return np.matmul(input, self.w.weights) + self.b.weights
