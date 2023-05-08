from typing import Tuple

from src.game import SnakeGame


def fitness_time_length_alive(game: SnakeGame) -> Tuple[float, float]:
    # Add survived time to fitness function
    fitness1 = game.snake1.lifetime
    fitness2 = game.snake2.lifetime

    # Add snakes length to fitness function
    fitness1 += (game.snake1.body.qsize() - 3) * 10
    fitness2 += (game.snake2.body.qsize() - 3) * 10
    
    # Add 50 points to fitness function of survivors
    if game.snake1.alive:
        fitness1 += 50
    if game.snake2.alive:
        fitness2 += 50
    
    return fitness1, fitness2
