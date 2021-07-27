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

import argparse
import sys



RANDOM_SEED = 41239678
RANDOM_STATE = np.random.RandomState(RANDOM_SEED)
GRID = gen.create_grid((10, 10), obstacle_prob=0.2, random_state=RANDOM_STATE)
G = gen.MDPGraph.from_grid(GRID, strat=gen.ActionStrategy.WRAP_NOOP_EFFECT)
# TODO: randomize? Add noise later?
G.R[G.R != 1] = -0.001
ENV = MDPGraphEnv(G)

def env_creator(_):
    return ENV # return an env instance

# TODO:
# - small negative rewards, when reward is 0?
# - random variation/noise of transition probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='test specified checkpoint')
    parser.add_argument('--iters', default=100, help='number of iters to train')
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'dqn'], help='which algorithm to train with')
    args = parser.parse_args()
    train = args.test is None
    iters = int(args.iters)
    algo = ppo if args.algo == 'ppo' else dqn
    algo_trainer = PPOTrainer if args.algo == 'ppo' else DQNTrainer
    register_env("gridworld", env_creator)

    config = algo.DEFAULT_CONFIG.copy()
    print(config)
    config.update({"env": "gridworld",
            'env_config':{'visualize':False},
            # "callbacks": {
            #     "on_train_result": on_train_result,
            # },

            'framework':'torch',
            "num_workers": 4,
            'model': {
                    "use_lstm": False,
            }
            })


    if train:
        tune.run(algo_trainer, config=config,
                checkpoint_freq = 10,
                name="gridworld",
                stop={'training_iteration': iters}
        ) 
    else:
        ray.init()
        agent = algo_trainer(config=config, env='gridworld')
        agent.restore(args.test)

        done = False
        ep_reward = 0
        obs = ENV.reset()
        print(obs)
        step_count = 0
        while not done:
                step_count += 1
                action = agent.compute_action(obs)
                print(action)
                obs, reward, done, _ = ENV.step(action)
                print(obs)
                ep_reward += reward
        print(obs, ep_reward, step_count)
        import code; code.interact(local=vars())

