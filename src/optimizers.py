from abc import ABC, abstractmethod
from typing import Type, Tuple, Callable
from heapq import heapify, heappush, heappop
import random
import time

from src.models import GameAgent
from src.game import CellState, SnakeGame, GameEndException

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def fit(self):
        pass


class EvolutionOptimizer(Optimizer):
    def __init__(
        self,
        agent_class: Type[GameAgent],
        fitness_function: Callable,
        mutation_probability: float = 0.01,
        mutation_scale: float = 1.0,
        population_size: int = 100,
        best_agents_cnt: int = 10,
        game_grid_shape: Tuple[int, int] = (10, 20),
        init_time: int = 100,
    ) -> None:
        super().__init__()

        if population_size % 2 == 1:
            raise ValueError('Population size must be even')
        if best_agents_cnt > population_size:
            raise ValueError('Top agents count cannot be greater than population size')

        self.agent_class = agent_class
        self.fitness_function = fitness_function
        self.mutation_probability = mutation_probability
        self.mutation_scale = mutation_scale
        self.population_size = population_size
        self.game_grid_shape = game_grid_shape
        self.init_time = init_time

        self.mean_fitness_history = []

        # Initialize top agents
        self.best_agents = [
            self.agent_class()
            for _ in range(best_agents_cnt)
        ]
        self.best_agents = [
            (0, id(agent), agent)
            for agent in self.best_agents
        ]
        heapify(self.best_agents)

    def __simulate_game(self, agent1: GameAgent, agent2: GameAgent) -> Tuple[float, float]:
        game = SnakeGame(self.game_grid_shape)
        time_left = self.init_time

        try:
            agent1_prev_len = game.snake1.body.qsize()
            agent2_prev_len = game.snake2.body.qsize()
            while time_left > 0:
                agent1_move = agent1.move(game, game.snake1) if game.snake1.alive else None
                agent2_move = agent2.move(game, game.snake2) if game.snake2.alive else None
                game.update(agent1_move, agent2_move)
                
                # Update time left
                if (game.snake1.body.qsize() > agent1_prev_len) or (game.snake2.body.qsize() > agent2_prev_len):
                    time_left = self.init_time
                else:
                    time_left -= 1
                
                # Update agents length
                agent1_prev_len = game.snake1.body.qsize()
                agent2_prev_len = game.snake2.body.qsize()
        except GameEndException:
            pass

        return self.fitness_function(game)

    def __create_agent(self) -> GameAgent:
        agent = self.agent_class()
        random_best_agent = self.best_agents[random.randint(0, len(self.best_agents) - 1)][2]
        for new_agent_param, best_agent_param in zip(agent.params(), random_best_agent.params()):
            # Copy weights from top agent
            new_agent_param.weights = best_agent_param.weights.copy()
            # Randomly mutate some weights
            mask = np.random.rand(*new_agent_param.weights.shape) <= self.mutation_probability
            new_agent_param.weights[mask] +=\
                np.random.normal(scale=self.mutation_scale, size=new_agent_param.weights[mask].shape)
        return agent

    def __fit_epoch(self):
        new_best_agents = []
        for _ in range(self.population_size // 2):
            agent1, agent2 = self.__create_agent(), self.__create_agent()
            fitness1, fitness2 = self.__simulate_game(agent1, agent2)
            
            heappush(new_best_agents, (fitness1, id(agent1), agent1))
            heappush(new_best_agents, (fitness2, id(agent2), agent2))

            while len(new_best_agents) > len(self.best_agents):
                heappop(new_best_agents)
        self.best_agents = new_best_agents

    def fit(self, num_epochs: int = 1):
        start = time.time()
        for i in range(num_epochs):
            epoch_start = time.time()
            self.__fit_epoch()

            mean_fitness = sum([agent[0] for agent in self.best_agents]) / len(self.best_agents)
            self.mean_fitness_history.append(mean_fitness)
            
            print(f'Epoch: {i+1:>4}, Mean fitness: {round(mean_fitness, 3):.3f}, Time: {round(time.time() - epoch_start, 2)}s')
        print(f'Total time: {round(time.time() - start, 2)}s')


class GeneticOptimizer(Optimizer):
    def __init__(
        self,
        agent_class: Type[GameAgent],
        fitness_function: Callable,
        mutation_probability: float = 0.01,
        mutation_scale: float = 1.0,
        crossing_probability: float = 0.2,
        population_size: int = 100,
        best_agents_cnt: int = 10,
        game_grid_shape: Tuple[int, int] = (10, 20),
        init_time: int = 100,
    ) -> None:
        super().__init__()

        if population_size % 2 == 1:
            raise ValueError('Population size must be even')
        if best_agents_cnt > population_size:
            raise ValueError('Top agents count cannot be greater than population size')

        self.agent_class = agent_class
        self.fitness_function = fitness_function
        self.mutation_probability = mutation_probability
        self.mutation_scale = mutation_scale
        self.crossing_probability = crossing_probability
        self.population_size = population_size
        self.game_grid_shape = game_grid_shape
        self.init_time = init_time

        self.mean_fitness_history = []

        # Initialize top agents
        self.best_agents = [
            self.agent_class()
            for _ in range(best_agents_cnt)
        ]
        self.best_agents = [
            (0, id(agent), agent)
            for agent in self.best_agents
        ]
        heapify(self.best_agents)

    def __simulate_game(self, agent1: GameAgent, agent2: GameAgent) -> Tuple[float, float]:
        game = SnakeGame(self.game_grid_shape)
        time_left = self.init_time

        try:
            agent1_prev_len = game.snake1.body.qsize()
            agent2_prev_len = game.snake2.body.qsize()
            while time_left > 0:
                agent1_move = agent1.move(game, game.snake1) if game.snake1.alive else None
                agent2_move = agent2.move(game, game.snake2) if game.snake2.alive else None
                game.update(agent1_move, agent2_move)
                
                # Update time left
                if (game.snake1.body.qsize() > agent1_prev_len) or (game.snake2.body.qsize() > agent2_prev_len):
                    time_left = self.init_time
                else:
                    time_left -= 1
                                
                # Update agents length
                agent1_prev_len = game.snake1.body.qsize()
                agent2_prev_len = game.snake2.body.qsize()
        except GameEndException:
            pass

        return self.fitness_function(game)

    def __create_agent(self) -> GameAgent:
        agent = self.agent_class()
        # Crossing
        if random.random() < self.crossing_probability:
            # Fill weights by crossing two random agents
            random_best_agent1 = self.best_agents[random.randint(0, len(self.best_agents) - 1)][2]
            random_best_agent2 = self.best_agents[random.randint(0, len(self.best_agents) - 1)][2]
            for new_agent_param, best_agent1_param, best_agent2_param in\
                zip(agent.params(), random_best_agent1.params(), random_best_agent2.params()):
                new_agent_param.weights = (best_agent1_param.weights + best_agent2_param.weights) / 2
        else:
            # Fill weights by copying
            random_best_agent = self.best_agents[random.randint(0, len(self.best_agents) - 1)][2]
            for new_agent_param, best_agent_param in zip(agent.params(), random_best_agent.params()):
                # Copy weights from random best agent
                new_agent_param.weights = best_agent_param.weights.copy()

        # Mutation
        for new_agent_param in agent.params():
            # Randomly mutate some weights
            mask = np.random.rand(*new_agent_param.weights.shape) <= self.mutation_probability
            new_agent_param.weights[mask] +=\
                np.random.normal(scale=self.mutation_scale, size=new_agent_param.weights[mask].shape)
        return agent

    def __fit_epoch(self):
        new_best_agents = []
        for _ in range(self.population_size // 2):
            agent1, agent2 = self.__create_agent(), self.__create_agent()
            fitness1, fitness2 = self.__simulate_game(agent1, agent2)
            
            heappush(new_best_agents, (fitness1, id(agent1), agent1))
            heappush(new_best_agents, (fitness2, id(agent2), agent2))

            while len(new_best_agents) > len(self.best_agents):
                heappop(new_best_agents)
        self.best_agents = new_best_agents

    def fit(self, num_epochs: int = 1):
        start = time.time()
        for i in range(num_epochs):
            epoch_start = time.time()
            self.__fit_epoch()

            mean_fitness = sum([agent[0] for agent in self.best_agents]) / len(self.best_agents)
            self.mean_fitness_history.append(mean_fitness)
            
            print(f'Epoch: {i+1:>4}, Mean fitness: {round(mean_fitness, 3):.3f}, Time: {round(time.time() - epoch_start, 2)}s')
        print(f'Total time: {round(time.time() - start, 2)}s')
