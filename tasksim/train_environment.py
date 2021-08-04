from ray import tune
import gym, ray
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
    
    def set_obs_size(self, obs_size):
        self.obs_size = obs_size
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
    
    # TODO: rotate?
    #def rotate(self, )
        
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
                    # TODO: scale by element or nah? Probably?
                    P[i, j] = max(0, P[i, j] + z)
                normed_row = sk_norm(np.array([P[i]]), norm='l1')[0]
                P[i, :] = normed_row
                out_a[i] = normed_row.copy()
            return G
        transition_random_state = self.coalesce_random(transition_random_state)
        if self.transition_noise > 0:
            G = add_noise(G, self.transition_noise, transition_random_state)
        env = MDPGraphEnv(graph=G, obs_size=self.obs_size, random_state=env_random_state)
        return env
    
    # Utility functions
    def row_col(self, state):
        _, width = self.grid.shape
        row = state // width
        col = state - self.width*row
        return row, col
    def flatten(self, row, col):
        _, width = self.grid.shape
        return row*width + col
    def coalesce_random(self, random_state=None):
        return np.random if random_state is None else random_state

RANDOM_SEED = 41239678
RANDOM_STATE = np.random.RandomState(RANDOM_SEED)
# GRID = gen.create_grid((15, 15), obstacle_prob=0.2, random_state=RANDOM_STATE)
# G = gen.MDPGraph.from_grid(GRID, strat=gen.ActionStrategy.WRAP_NOOP_EFFECT)
# # TODO: randomize? Add noise later?
# G.R[G.R != 1] = -0.001
# ENV = MDPGraphEnv(G, obs_size=3)
ENV = EnvironmentBuilder((15, 15)).set_obstacles(obstacle_prob=0.2, obstacle_random_state=RANDOM_STATE).build()

def env_creator(_):
    return ENV # return an env instance

# return mean, std, list and such
def test_env(env: MDPGraphEnv, agent):
    total_steps = 0
    total_reward = 0
    total_eps = 0
    grid = env.grid
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            grid_entry = grid[i, j]
            if grid_entry != 0:
                continue
            total_eps += 1
            obs = env.reset(i*cols + j)
            done = False
            ep_reward = 0
            state = agent.get_policy().model.get_initial_state()
            step_count = 0
            while not done:
                step_count += 1
                # TODO: is this stochastic? Can we make it determinstic?
                action, state, _ = agent.compute_action(observation=obs, prev_action=0, prev_reward=0, state=state)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
            total_steps += step_count
            total_reward += ep_reward
    return total_steps/total_eps, total_reward/total_eps

# TODO:
# - small negative rewards, when reward is 0?
# - random variation/noise of transition probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='test specified checkpoint')
    parser.add_argument('--last', help='use last training session to resume training [NOT SUPPORTED] or test', action='store_true')
    parser.add_argument('--iters', default=100, help='number of iters to train')
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'dqn'], help='which algorithm to train with')
    args = parser.parse_args()
    train = args.test is None
    iters = int(args.iters)
    last = args.last
    algo = ppo if args.algo == 'ppo' else dqn
    algo_trainer = PPOTrainer if args.algo == 'ppo' else DQNTrainer
    register_env("gridworld", env_creator)

    if last:
        base_dir = os.path.expanduser('~/ray_results/gridworld')
        def latest_file(path, dir=True):
            files = [os.path.join(path, f) for f in os.listdir(path)]
            return max([f for f in files if (os.path.isdir(f) if dir else os.path.isfile(f))], key=os.path.getctime)
        latest_run = latest_file(base_dir)
        latest_checkpoint_dir = latest_file(latest_run)
        latest_checkpoint_files = [f for f in os.listdir(latest_checkpoint_dir) if 'checkpoint-' in f and '.' not in f]
        assert len(latest_checkpoint_files) == 1, 'Could not find latest checkpoint file'
        last_path = os.path.join(latest_checkpoint_dir, latest_checkpoint_files[0])
    
    config = algo.DEFAULT_CONFIG.copy()
    print(config)
    config.update({"env": "gridworld",
            'env_config':{'visualize':False},
            # "callbacks": {
            #     "on_train_result": on_train_result,
            # },

            'framework':'torch',
            "num_workers": 8,
            'model': {
                    #TODO: consider turning to True
                    "use_lstm": True,
            }
            })


    if train:
        results = tune.run(algo_trainer, config=config,
                checkpoint_freq = 10,
                name="gridworld",
                # TODO: use stopping condition of episodes_total for transfer learning?
                stop={'training_iteration': iters}
        ) 
        metric='training_iteration'
        mode='max'
        path = results.get_best_checkpoint(results.get_best_logdir(metric, mode), metric, mode)
        print(path)
    else:
        ray.init()
        agent = algo_trainer(config=config, env='gridworld')
        restore_path = args.test if not last else last_path
        print('Restoring from', restore_path)
        agent.restore(args.test if not last else last_path)

        print(test_env(ENV, agent))
        import code; code.interact(local=vars())

