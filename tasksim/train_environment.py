from ray import tune
import gym, ray
from gym import spaces
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import dqn
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.registry import register_env
import tasksim
from tasksim.environment import MDPGraphEnv
from ray.tune.schedulers import PopulationBasedTraining
import tasksim.gridworld_generator as gen
import numpy as np
from sklearn.preprocessing import normalize as sk_norm

import argparse
import os
import sys
from collections import deque
from copy import deepcopy
import json
from datetime import datetime

class EnvironmentBuilder:
    def __init__(self, shape):
        self.shape = shape
        rows, cols = shape
        self.grid = np.zeros((rows, cols))
        center_row, center_col = rows//2, cols//2
        self.goal_locations = [self.flatten(center_row, center_col)]
        self.success_prob = 0.9
        self.transition_noise = 0
        self.obstacle_locations = []
        self.step_reward = 0
        self.strat = gen.ActionStrategy.WRAP_NOOP_EFFECT
        self.obs_size = 7
        self.fixed_start = None
        self.do_render = True
    
    def set_obs_size(self, obs_size):
        self.obs_size = obs_size
        return self
    def set_do_render(self, do_render):
        self.do_render = do_render
        return self
    def set_strat(self, strat):
        self.strat = strat
        return self
    def set_transition_noise(self, transition_noise):
        self.transition_noise = transition_noise
        return self
    def set_success_prob(self, success_prob):
        self.success_prob = success_prob
        return self
    def set_step_reward(self, step_reward):
        self.step_reward = step_reward
        return self
    def set_goals(self, goal_locations):
        self.goal_locations = [g for g in goal_locations]
        return self
    def set_obstacles(self, obstacle_locations=None, obstacle_prob=None, obstacle_random_state=None):
        if obstacle_prob is None:
            obstacle_prob = 0
        if obstacle_locations is None:
            obstacle_locations = []
        obstacle_random_state = self.coalesce_random(obstacle_random_state)
        self.obstacle_locations = [o for o in obstacle_locations]
        # populate with random probs
        rows, cols = self.grid.shape
        for i in range(rows):
            for j in range(cols):
                if self.grid[i, j] != 0:
                    continue
                if obstacle_random_state.rand() < obstacle_prob:
                    self.obstacle_locations.append(self.flatten(i, j))
        return self
    
    def set_fixed_start(self, fixed_start=None):
        self.fixed_start=fixed_start
        return self
        
    def build(self, transition_random_state=None, env_random_state=None):
        for o in self.obstacle_locations:
            row, col = self.row_col(o)
            self.grid[row, col] = 1
        for g in self.goal_locations:
            row, col = self.row_col(g)
            self.grid[row, col] = 2

        G = gen.MDPGraph.from_grid(grid=self.grid, success_prob=self.success_prob, strat=self.strat)
        G.R[G.R != 1] = self.step_reward
        def add_noise(G, percent, random_state):
            G = G.copy()
            P = G.P
            out_a = G.out_a
            n_actions, n_states = P.shape
            for i in range(n_actions):
                for j in range(n_states):
                    if P[i, j] <= 0:
                        continue
                    # Between [0, 1)
                    noise = random_state.rand()
                    # Between [-percent, +percent)
                    z = -percent + 2*percent*noise
                    P[i, j] = max(0, P[i, j] + z)
                normed_row = sk_norm(np.array([P[i]]), norm='l1')[0]
                P[i, :] = normed_row
                out_a[i] = normed_row.copy()
            return G
        transition_random_state = self.coalesce_random(transition_random_state)
        if self.transition_noise > 0:
            G = add_noise(G, self.transition_noise, transition_random_state)
        env = MDPGraphEnv(graph=G, obs_size=self.obs_size, do_render=self.do_render, random_state=env_random_state, fixed_start=self.fixed_start)
        return env
    
    # Utility functions
    def row_col(self, state):
        _, width = self.grid.shape
        row = state // width
        col = state - width*row
        return row, col
    def flatten(self, row, col):
        _, width = self.grid.shape
        return row*width + col
    def coalesce_random(self, random_state=None):
        return np.random if random_state is None else random_state