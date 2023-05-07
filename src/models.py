from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np

from src.layers import Parameter, Layer, LinearLayer, ReLU
from src.game import Direction, CellState, SnakeGame


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


class GameAgent(Model):
    @abstractmethod
    def extract_input(self, game: SnakeGame, snake_num: int) -> np.array:
        pass

    @abstractmethod
    def choose_direction(self, *args: Any, **kwargs: Any) -> Direction:
        pass

    @abstractmethod
    def move(self, game: SnakeGame, snake_num: int) -> Direction:
        pass


class SnakeAgent(GameAgent):
    def __init__(self) -> None:
        super().__init__()
        self.input_layer = LinearLayer(16, 18)
        self.hidden_layer1 = LinearLayer(18, 10)
        self.output_layer = LinearLayer(10, 4)
        self.relu = ReLU()
    
    def forward(self, input: np.array) -> np.array:
        output = self.relu(self.input_layer(input))
        output = self.relu(self.hidden_layer1(output))
        output = self.output_layer(output)
        output = np.argmax(output, axis=1)
        return output

    def extract_input(self, game: SnakeGame, snake_num: int) -> np.array:
        def find_closest_in_grid(
            grid: np.array,
            pos: Tuple[int, int],
            dir: Direction,
            target: int,
        ) -> int:
            if dir == Direction.UP:
                dir = (-1, 0)
            elif dir == Direction.DOWN:
                dir = (1, 0)
            elif dir == Direction.LEFT:
                dir = (0, -1)
            elif dir == Direction.RIGHT:
                dir = (0, 1)

            start_pos = pos
            while 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1] and grid[pos] != target:
                pos = (pos[0] + dir[0], pos[1] + dir[1])
            
            if 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1] and grid[pos] == target:
                return max([abs(pos[0] - start_pos[0]) - 1, abs(pos[1] - start_pos[1]) - 1])
            else:
                return -1

        input = np.zeros((1, 16)) - 1
        snake = game.snake1 if snake_num == CellState.SNAKE1.value else game.snake2

        # Calculate distances from head to walls
        input[0, 0] = snake.queue[-1][0]                      # Distance head to upper wall
        input[0, 1] = game.grid.shape[0] - snake.queue[-1][0] # Distnace head to lower wall
        input[0, 2] = snake.queue[-1][1]                      # Distnace head to left wall
        input[0, 3] = game.grid.shape[1] - snake.queue[-1][1] # Distnace head to right wall

        # Calculate distance from head to food
        food_pos = np.where(game.grid == CellState.FOOD.value)
        food_pos = (food_pos[0][0], food_pos[1][0])
        input[0, 4] = snake.queue[-1][0] - food_pos[0] # Y distance head to food (UP direction)
        input[0, 5] = snake.queue[-1][1] - food_pos[1]  # X distance head to food (LEFT direction)

        # Calculate distance from head to tail
        input[0, 6] = snake.queue[-1][0] - snake.queue[0][0] # Y distance from head to tail (UP direction)
        input[0, 7] = snake.queue[-1][1] - snake.queue[0][1] # X distance from head to tail (LEFT direction)

        # Calculate distances to self body
        input[0, 8] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.UP, snake_num)
        input[0, 9] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.DOWN, snake_num)
        input[0, 10] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.LEFT, snake_num)
        input[0, 11] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.RIGHT, snake_num)

        # Calculate distances to another body
        other_snake_num = CellState.SNAKE1.value if snake_num == CellState.SNAKE2.value else CellState.SNAKE2.value
        input[0, 12] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.RIGHT, other_snake_num)
        input[0, 13] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.RIGHT, other_snake_num)
        input[0, 14] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.RIGHT, other_snake_num)
        input[0, 15] = find_closest_in_grid(game.grid, snake.queue[-1], Direction.RIGHT, other_snake_num)

        return input

    def choose_direction(self, output: np.array) -> Direction:
        return list(Direction)[output[0]]

    def move(self, game: SnakeGame, snake_num: int) -> Direction:
        input = self.extract_input(game, snake_num)
        output = self.forward(input)
        direction = self.choose_direction(output)
        return direction
