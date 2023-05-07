from abc import ABC, abstractmethod
from typing import Type, Tuple, Callable
from heapq import heapify, heappushpop
import random
import time

from src.models import GameAgent
from src.game import CellState, SnakeGame, GameEndException

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def fit(self):
        pass


class SimpleMutationOptimizer(Optimizer):
    def __init__(
        self,
        agent_class: Type[GameAgent],
        fitness_function: Callable,
        mutation_probability: float = 0.01,
        mutation_scale: float = 1.0,
        population_size: int = 100,
        top_agents_cnt: int = 10,
        game_grid_shape: Tuple[int, int] = (10, 20),
        init_steps: int = 100,
    ) -> None:
        super().__init__()

        if population_size % 2 == 1:
            raise ValueError('Population size must be even')
        if top_agents_cnt > population_size:
            raise ValueError('Top agents count cannot be greater than population size')

        self.agent_class = agent_class
        self.fitness_function = fitness_function
        self.mutation_probability = mutation_probability
        self.mutation_scale = mutation_scale
        self.population_size = population_size
        self.game_grid_shape = game_grid_shape
        self.init_steps = init_steps

        # Initialize top agents
        self.top_agents = [
            self.agent_class()
            for _ in range(top_agents_cnt)
        ]
        self.top_agents = [
            (0, id(agent), agent)
            for agent in self.top_agents
        ]
        heapify(self.top_agents)

    def __simulate_game(self, agent1: GameAgent, agent2: GameAgent) -> Tuple[float, float]:
        game = SnakeGame(self.game_grid_shape)
        steps_left = self.init_steps

        try:
            agent1_prev_len = game.snake1.qsize()
            agent2_prev_len = game.snake2.qsize()
            while steps_left > 0:
                game.update(
                    agent1.move(game, CellState.SNAKE1.value),
                    agent2.move(game, CellState.SNAKE2.value),
                )
                
                steps_left -= 1
                # Add 10 steps if any snake ate food
                steps_left += 10 * (game.snake1.qsize() - agent1_prev_len + game.snake2.qsize() - agent2_prev_len)
                
                agent1_prev_len = game.snake1.qsize()
                agent2_prev_len = game.snake2.qsize()
        except GameEndException:
            pass

        return self.fitness_function(game)

    def __create_agent(self) -> GameAgent:
        agent = self.agent_class()
        top_agent = self.top_agents[random.randint(0, len(self.top_agents) - 1)][2]
        for new_agent_param, top_agent_param in zip(agent.params(), top_agent.params()):
            # Copy weights from top agent
            new_agent_param.weights = top_agent_param.weights
            # Randomly mutate some weights
            mask = np.random.rand(*new_agent_param.weights.shape) <= self.mutation_probability
            new_agent_param.weights[mask] += (np.random.rand(*new_agent_param.weights[mask].shape) - 0.5) * 2 * self.mutation_scale
        return agent

    def __fit_epoch(self):
        for _ in range(self.population_size // 2):
            agent1, agent2 = self.__create_agent(), self.__create_agent()
            fitness1, fitness2 = self.__simulate_game(agent1, agent2)
            heappushpop(self.top_agents, (fitness1, id(agent1), agent1))
            heappushpop(self.top_agents, (fitness2, id(agent2), agent2))

    def fit(self, num_epochs: int = 1):
        for i in range(num_epochs):
            start = time.time()
            self.__fit_epoch()
            mean_fitness = sum([agent[0] for agent in self.top_agents]) / len(self.top_agents)
            print(f'Epoch: {i+1:>4}, Mean fitness: {round(mean_fitness, 3)}, Time: {round(time.time() - start, 2)}s')


class GeneticOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self):
        pass


class EvolutionOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self):
        pass
