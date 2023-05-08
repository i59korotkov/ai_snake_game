from typing import Tuple, Optional
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


def reverse_direction(dir: Direction) -> Direction:
    if dir == Direction.UP:
        return Direction.DOWN
    elif dir == Direction.DOWN:
        return Direction.UP
    elif dir == Direction.LEFT:
        return Direction.RIGHT
    elif dir == Direction.RIGHT:
        return Direction.LEFT


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
    DARK_BLUE = (0, 0, 128)
    DARK_RED = (128, 0, 0)


class SnakeInfo:
    def __init__(
        self,
        body: Queue,
        num: int,
        alive: bool = True,
        lifetime: int = 0,
    ) -> None:
        self.body = body
        self.num = num
        self.alive = alive
        self.lifetime = lifetime


class SnakeGame:
    def __init__(self, grid_shape: Tuple[int, int] = (10, 10)) -> None:
        if len(grid_shape) != 2 or grid_shape[0] < 5 or grid_shape[1] < 5:
            raise ValueError('Invalid grid shape')

        self.grid = np.zeros(grid_shape, dtype=int)
        self.time = 0
        self._init_snakes()
        self._generate_food()
    
    def _init_snakes(self):
        self.snake1 = SnakeInfo(Queue(), CellState.SNAKE1.value)
        self.snake2 = SnakeInfo(Queue(), CellState.SNAKE2.value)

        # Init snakes on random Y positions
        y_pos1 = random.randint(1, self.grid.shape[0] - 2)
        y_pos2 = random.randint(1, self.grid.shape[0] - 2)
        for x in range(3):
            self.snake1.body.put((y_pos1, 1 + x))
            self.snake2.body.put((y_pos2, self.grid.shape[1] - 2 - x))
        
        for cell in self.snake1.body.queue:
            self.grid[cell] = CellState.SNAKE1.value
        for cell in self.snake2.body.queue:
            self.grid[cell] = CellState.SNAKE2.value

    def _move_snake(self, snake: SnakeInfo, dir: Direction):
        if not snake.alive:
            return

        snake.lifetime += 1

        # Get new head position
        if dir == Direction.UP:
            head = snake.body.queue[-1]
            head = (head[0] - 1, head[1])
        elif dir == Direction.DOWN:
            head = snake.body.queue[-1]
            head = (head[0] + 1, head[1])
        elif dir == Direction.LEFT:
            head = snake.body.queue[-1]
            head = (head[0], head[1] - 1)
        elif dir == Direction.RIGHT:
            head = snake.body.queue[-1]
            head = (head[0], head[1] + 1)

        if head == snake.body.queue[-2]:
            # Reverse direction, if the movement was inside the snake body
            self._move_snake(snake, reverse_direction(dir))
            return
        
        if 0 <= head[0] < self.grid.shape[0] and 0 <= head[1] < self.grid.shape[1]:
            if self.grid[head] == CellState.EMPTY.value:
                # If new head cell is empty
                tail = snake.body.get()
                snake.body.put(head)
                self.grid[tail] = 0
                self.grid[head] = snake.num
            elif self.grid[head] == CellState.FOOD.value:
                # If new head cell has food in it
                snake.body.put(head)
                self.grid[head] = snake.num
            else:
                # If new head cell is body of a snake
                snake.alive = False
        else:
            # If new head cell is outside of the grid
            snake.alive = False

    def _generate_food(self):
        empty_cells = np.where(self.grid == CellState.EMPTY.value)
        random_index = np.random.randint(len(empty_cells[0]), size=1)[0]
        self.grid[empty_cells[0][random_index], empty_cells[1][random_index]] = CellState.FOOD.value

    def update(self, snake1_move_dir: Optional[Direction], snake2_move_dir: Optional[Direction]):
        # Randomly select which snake moves first
        if random.random() > 0.5:
            self._move_snake(self.snake1, snake1_move_dir)
            self._move_snake(self.snake2, snake2_move_dir)
        else:
            self._move_snake(self.snake2, snake2_move_dir)
            self._move_snake(self.snake1, snake1_move_dir)

        if not (self.grid == CellState.FOOD.value).any():
            # Generate food if it was eaten
            self._generate_food()

        self.time += 1
        if not self.snake1.alive and not self.snake2.alive:
            # End game if both snakes die
            raise GameEndException()
    
    def get_image(self) -> np.array:
        image = np.zeros((*self.grid.shape, 3), dtype=int)
        image[np.where(self.grid == CellState.EMPTY.value)] = Colors.BLACK.value
        if self.snake1.body.qsize() > 0:
            image[np.where(self.grid == CellState.SNAKE1.value)] = Colors.BLUE.value
            image[self.snake1.body.queue[-1]] = Colors.DARK_BLUE.value
        if self.snake2.body.qsize() > 0:
            image[np.where(self.grid == CellState.SNAKE2.value)] = Colors.RED.value
            image[self.snake2.body.queue[-1]] = Colors.DARK_RED.value
        image[np.where(self.grid == CellState.FOOD.value)] = Colors.WHITE.value
        return image
