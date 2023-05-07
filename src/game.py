from typing import Tuple
from enum import Enum
from queue import Queue
import random

import numpy as np


class GameEndException(Exception):
    pass


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellState(Enum):
    EMPTY = 0
    SNAKE1 = 1
    SNAKE2 = 2
    FOOD = 3


class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)


def reverse_direction(dir: Direction) -> Direction:
    if dir == Direction.UP:
        return Direction.DOWN
    elif dir == Direction.DOWN:
        return Direction.UP
    elif dir == Direction.LEFT:
        return Direction.RIGHT
    elif dir == Direction.RIGHT:
        return Direction.LEFT


class SnakeGame:
    def __init__(self, grid_shape: Tuple[int, int] = (10, 20)) -> None:
        if len(grid_shape) != 2 or grid_shape[0] < 5 or grid_shape[1] < 5:
            raise ValueError('Invalid grid shape')

        self.grid = np.zeros(grid_shape, dtype=int)
        self.step = 0
        self.__init_snakes()
        self.__generate_food()
    
    def __init_snakes(self):
        self.snake1 = Queue()
        self.snake2 = Queue()
        self.snake1_is_alive = True
        self.snake2_is_alive = True

        for x in range(3):
            self.snake1.put((1, 1 + x))
            self.snake2.put((self.grid.shape[0] - 2, self.grid.shape[1] - 2 - x))
        
        for cell in self.snake1.queue:
            self.grid[cell] = CellState.SNAKE1.value
        for cell in self.snake2.queue:
            self.grid[cell] = CellState.SNAKE2.value

    def __move_snake(self, snake: Queue, dir: Direction, snake_num: int) -> bool:
        # Get new head position
        if dir == Direction.UP:
            head = snake.queue[-1]
            head = (head[0] - 1, head[1])
        elif dir == Direction.DOWN:
            head = snake.queue[-1]
            head = (head[0] + 1, head[1])
        elif dir == Direction.LEFT:
            head = snake.queue[-1]
            head = (head[0], head[1] - 1)
        elif dir == Direction.RIGHT:
            head = snake.queue[-1]
            head = (head[0], head[1] + 1)

        if head == snake.queue[-2]:
            # Reverse direction, if the movement was inside the snake body
            return self.__move_snake(snake, reverse_direction(dir), snake_num)
        
        if 0 <= head[0] < self.grid.shape[0] and 0 <= head[1] < self.grid.shape[1]:
            if self.grid[head] == CellState.EMPTY.value:
                # If new head cell is empty
                tail = snake.get()
                snake.put(head)
                self.grid[tail] = 0
                self.grid[head] = snake_num
                return True
            elif self.grid[head] == CellState.FOOD.value:
                # If new head cell has food in it
                snake.put(head)
                self.grid[head] = snake_num
                return True
            else:
                # If new head cell is body of a snake
                return False
        else:
            # If new head cell is outside of the grid
            return False

    def __generate_food(self):
        empty_cells = np.where(self.grid == CellState.EMPTY.value)
        random_index = np.random.randint(len(empty_cells[0]), size=1)[0]
        self.grid[empty_cells[0][random_index], empty_cells[1][random_index]] = CellState.FOOD.value

    def update(self, snake1_move_dir: Direction, snake2_move_dir: Direction):
        # Randomly select which snake moves first
        if random.random() > 0.5:
            self.snake1_is_alive = self.__move_snake(self.snake1, snake1_move_dir, CellState.SNAKE1.value)
            self.snake2_is_alive = self.__move_snake(self.snake2, snake2_move_dir, CellState.SNAKE2.value)
        else:
            self.snake2_is_alive = self.__move_snake(self.snake2, snake2_move_dir, CellState.SNAKE2.value)
            self.snake1_is_alive = self.__move_snake(self.snake1, snake1_move_dir, CellState.SNAKE1.value)

        if not (self.grid == CellState.FOOD.value).any():
            # Generate food if it was eaten
            self.__generate_food()

        self.step += 1
        if not self.snake1_is_alive or not self.snake2_is_alive:
            # End game if any snake died
            raise GameEndException()
    
    def get_image(self) -> np.array:
        image = np.zeros((*self.grid.shape, 3), dtype=int)
        image[np.where(self.grid == CellState.EMPTY.value)] = Colors.BLACK.value
        image[np.where(self.grid == CellState.SNAKE1.value)] = Colors.BLUE.value
        image[np.where(self.grid == CellState.SNAKE2.value)] = Colors.RED.value
        image[np.where(self.grid == CellState.FOOD.value)] = Colors.WHITE.value
        return image
