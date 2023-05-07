from typing import Tuple

from src.game import SnakeGame


def fitness_steps_length_alive(game: SnakeGame) -> Tuple[float, float]:
    # Add survived steps to fitness function
    fitness1 = game.step / 10
    fitness2 = game.step / 10

    # Add snakes length to fitness function
    fitness1 += game.snake1.qsize() 
    fitness2 += game.snake2.qsize()
    
    # Add 10 points to fitness function of survivors
    if game.snake1_is_alive:
        fitness1 += 10
    if game.snake2_is_alive:
        fitness2 += 10
    
    return fitness1, fitness2


def fitness_steps_length(game: SnakeGame) -> Tuple[float, float]:
    # Add survived steps to fitness function
    fitness1 = game.step / 10
    fitness2 = game.step / 10

    # Add snakes length to fitness function
    fitness1 += game.snake1.qsize() 
    fitness2 += game.snake2.qsize()
    
    return fitness1, fitness2
