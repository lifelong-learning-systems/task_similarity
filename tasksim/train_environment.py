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

from experiments.pipeline_utilities import progress_bar

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
        col = state - width*row
        return row, col
    def flatten(self, row, col):
        _, width = self.grid.shape
        return row*width + col
    def coalesce_random(self, random_state=None):
        return np.random if random_state is None else random_state

# return mean, std, list and such
def test_env(env: MDPGraphEnv, agent, n=1000, num_sims=100, agent_max_step=1000, lstm=False, sim_random_state=None):
    strat = env.graph.strat
    graph = env.graph
    grid = graph.grid
    rows, cols = grid.shape
    height, width = grid.shape
    optimal_action_grid = np.nan * np.zeros(graph.grid.shape)
    optimal_path_len = np.zeros(graph.grid.shape)
    sim_random_state = np.random if sim_random_state is None else sim_random_state

    def compute_optimal_path(start_state):
        graph = env.graph
        grid = graph.grid
        q = deque() # q.append(x) -> pushes on to the right side, q.popleft() -> removes and returns from left
        q.append((start_state, []))
        found = None
        visited = set()
        while len(q):
            cur_state, cur_path = q.popleft()
            if cur_state in visited:
                continue
            visited.add(cur_state)
            # left, right, up, down, noop
            adjacent = gen.MDPGraph.get_valid_adjacent(cur_state, grid, strat)
            filtered = [a for a in adjacent if a is not None]
            out_actions = graph.out_s[cur_state]
            assert len(out_actions) == len(filtered), 'darn, something went wrong'
            for id, f in enumerate(filtered):
                if f == cur_state:
                    continue
                looped = False
                for path_state, _ in cur_path:
                    if path_state == f:
                        looped = True
                        break
                if looped:
                    continue
                action = out_actions[id]
                row = f // width
                col = f - width*row
                next_path = [c for c in cur_path]
                next_path.append((cur_state, action))
                if grid[row, col] == 2:
                    found = (f, next_path)
                    break
                else:
                    q.append((f, next_path))
            if found is not None:
                break
        assert found is not None, 'Could not find a path from start state to a goal'
        _, goal_path = found
        # goal_path is an optimal deterministic path from start_state to a goal_state
        # graph.P: Actions - Distribution among States
        for state, action in goal_path:
            row = state // width
            col = state - width*row
            entry = optimal_action_grid[row, col]
            if np.isnan(entry):
                optimal_action_grid[row, col] = action
            elif entry != action:
                print('WARNING: found mismatched optimal entry???')
                continue
        return len(goal_path)

    # 1. for every valid grid state, figure out the optimal action to take
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                continue
            path_len = compute_optimal_path(i*cols + j)
            optimal_path_len[i, j] = path_len

    # cached_optimal_steps = {}
    # for i in range(rows):
    #     for j in range(cols):
    #         if grid[i, j] != 0:
    #             continue
    #         base_state = i*cols + j
    #         sim_steps = 0
    #         for _ in range(num_sims):
    #             cur_state = base_state
    #             done = False
    #             while True:
    #                 row = cur_state // width
    #                 col = cur_state - width*row
    #                 if grid[row, col] == 2:
    #                     break
    #                 optimal_action = optimal_action_grid[row, col]
    #                 assert not np.isnan(optimal_action), 'No action found'
    #                 transitions = graph.P[int(optimal_action)]
    #                 indices = np.array([i for i in range(len(transitions))])
    #                 cur_state = sim_random_state.choice(indices, p=transitions)
    #                 sim_steps += 1
    #         sim_avg = sim_steps / num_sims
    #         cached_optimal_steps[base_state] = sim_avg
    performances = []
    episodes = list(range(n))
    optimal_step_list = []
    agent_step_list = []
    optimal_path_len = optimal_path_len.flatten()
    for _ in progress_bar(episodes, prefix='Test progress:', suffix='Complete'):
        obs = env.reset()
        optimal_steps = optimal_path_len[env.state]
        optimal_step_list.append(optimal_steps)

        done = False
        ep_reward = 0
        reward = 0
        action = 0
        if lstm:
            state = agent.get_policy().model.get_initial_state()
        step_count = 0
        while not done:
            step_count += 1
            if step_count >= agent_max_step:
                break
            # TODO: is this stochastic? Can we make it determinstic?
            if lstm:
                action, state, _ = agent.compute_action(observation=obs, prev_action=action, prev_reward=reward, state=state)
            else:
                action = agent.compute_action(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        agent_step_list.append(step_count)
        performance = optimal_steps / step_count if done else 0
        performances.append(performance)
    mean_performance = np.mean(performances)
    return mean_performance, np.mean(optimal_step_list), np.mean(agent_step_list)

# TODO:
# - small negative rewards, when reward is 0?
# - random variation/noise of transition probs

def create_envs():
    envs = []
    base_seed = 41239678
    transition_seed = 94619456
    OBS_SIZE = 17
    # ENV 0
    base_env = EnvironmentBuilder((15, 15)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .build()
    envs.append(base_env)

    # ENV 1
    noisy_env = EnvironmentBuilder((15, 15)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .set_transition_noise(0.35) \
            .build(transition_random_state=np.random.RandomState(transition_seed))
    envs.append(noisy_env)

    # ENV 2
    small_env = EnvironmentBuilder((11, 11)) \
            .set_obstacles(obstacle_prob=0.1, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .build()
    envs.append(small_env)

    # ENV 3
    goal_change_env = EnvironmentBuilder((15, 15)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .set_goals([15*13+13]) \
            .build()
    envs.append(goal_change_env)

    # ENV 4
    multi_goal_env = EnvironmentBuilder((15, 15)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .set_goals([15*7+7, 15*13+13, 15*1+1, 15*13+1, 15*1+13]) \
            .build()
    envs.append(multi_goal_env)

    # ENV 5
    goal_change2_env = EnvironmentBuilder((15, 15)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(OBS_SIZE) \
            .set_goals([15*7+8]) \
            .build()
    envs.append(goal_change2_env)

    return envs

def forward_transfer(agent, env, performance_goal):

    # a. get the goal performance from each STE agent
    # b. 

    # 1. train agent until performance_goal met 
    # 2. ???
    # 3. return # of timesteps
    pass

if __name__ == '__main__':
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    num_print_decimals = 5
    np.set_printoptions(linewidth=200, precision=num_print_decimals, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help='test instead of train the specified paths', action='store_true')
    parser.add_argument('--paths', default='agent_paths/ppo_3x3.json', help='meta folder for saving/restoring agents')
    parser.add_argument('--last', help='use last training session to resume training [NOT SUPPORTED] or test', action='store_true')
    parser.add_argument('--lstm', help='use lstm', action='store_true')
    parser.add_argument('--iters', default=100, help='number of iters to train')
    parser.add_argument('--episodes', default=10000, help='number of episodes to train')
    parser.add_argument('--steps', default=100000, help='number of timesteps to train')
    parser.add_argument('--kl', default=0.01, help='kl value to train to')
    parser.add_argument('--env', default=0, help='which env id to use')
    #parser.add_argument('--obs', default=3, help='observation size')
    parser.add_argument('--algo', default='ppo', choices=['ppo', 'dqn'], help='which algorithm to train with')
    args = parser.parse_args()
    train = not args.test
    iters = int(args.iters)
    episodes = int(args.episodes)
    timesteps = int(args.steps)
    kl_stop = float(args.kl)
    last = args.last
    lstm = args.lstm
    agent_paths = args.paths
    algo = ppo if args.algo == 'ppo' else dqn
    env_id = int(args.env)
    #env_obs_size = int(args.obs)
    algo_trainer = PPOTrainer if args.algo == 'ppo' else DQNTrainer

    # TODO: pass in obs_size here maybe
    envs = create_envs()
    def env_creator(config):
        return envs[config['env_id']]
    register_env("gridworld", env_creator)


    if last:
        base_dir = os.path.expanduser('~/ray_results/gridworld')
        def latest_file(path, dir=True):
            files = [os.path.join(path, f) for f in os.listdir(path)]
            dirs = [f for f in files if (os.path.isdir(f) if dir else os.path.isfile(f))]
            dirs.sort(key=os.path.getctime)
            return dirs[-1]
        latest_run = latest_file(base_dir)
        latest_checkpoint_dir = latest_file(latest_run)
        latest_checkpoint_files = [f for f in os.listdir(latest_checkpoint_dir) if 'checkpoint-' in f and '.' not in f]
        assert len(latest_checkpoint_files) == 1, 'Could not find latest checkpoint file'
        last_path = os.path.join(latest_checkpoint_dir, latest_checkpoint_files[0])
    
    config = algo.DEFAULT_CONFIG.copy()
    # for env in envs:
    #     env.obs_size = env_obs_size
    #     env.observation_space = spaces.Box(low=0, high=4, shape=(env_obs_size, env_obs_size), dtype=np.uint8)
    env_obs_size = envs[env_id].obs_size
    obs_size_str = f'{env_obs_size}x{env_obs_size}'
    config.update({"env": "gridworld",
                   'env_config':{'env_id': env_id, 'trial_name': f'{args.algo.upper()}_{"" if not lstm else "lstm_"}{obs_size_str}_gridworld_env-{env_id}'},
                   'framework':'torch',
                   "num_workers": 8,
                   'model': {
                        "use_lstm": lstm,
                        # TODO: add CNN? Make own model and just pass in? Reduce even further...
                        'fcnet_hiddens': [32, 32]
                       }
                    })
    if algo == dqn:
        learning_start = 10000
        config.update({
            'learning_starts': learning_start,
            'timesteps_per_iteration': learning_start,
            #'replay_sequence_length': 1,
            #'prioritized_replay': False
        })
        config['exploration_config']['epsilon_timesteps'] = learning_start*10
    print(config)

    def trial_name_string(_, config):
        return config['env_config']['trial_name']

    import torch
    from torch.nn.functional import softmax
    action_dist = lambda x, obs: x.get_policy().compute_single_action(obs)[2]['action_dist_inputs']
    def get_action(agent, obs, deterministic=False):
        dist = action_dist(agent, obs)
        dist = np.array(softmax(torch.FloatTensor(dist)))
        return np.argmax(dist) if deterministic else np.random.choice([i for i in range(len(dist))], p=dist)

    if train:
        results = tune.run(algo_trainer, config=config,
                checkpoint_freq = 1,
                name="gridworld",
                trial_name_creator=lambda trial: trial_name_string(trial, config),
                # TODO: use stopping condition of episodes_total for transfer learning?
                #stop={'training_iteration': iters}
                stop={'agent_timesteps_total': timesteps}#, 'kl': kl_stop}
        ) 
        metric='training_iteration'
        mode='max'
        path = results.get_best_checkpoint(results.get_best_logdir(metric, mode), metric, mode)
        # TODO: change save path? For easy access, e.g. env_id + algo + config? idk: Maybe key value, like "env_id-algo:<path>"
        # TODO: save training config as well? Save obs_size and LSTM or not at least...algo as well too
        print(path)
        if os.path.exists(agent_paths):
            with open(agent_paths, 'r') as f:
                path_obj = json.load(f)
        else:
            path_obj = {}
        path_obj[str(env_id)] = path
        with open(agent_paths, 'w') as f:
            json.dump(path_obj, f)
    else:
        ray.init()
        if os.path.exists(agent_paths) and not last:
            with open(agent_paths, 'r') as f:
                path_obj = json.load(f)
            fixed_path_obj = {int(k): v for k, v in path_obj.items()}
            stops = [{'episode_reward_mean': x} for x in [.9909, .9884, .9942, .9908, .9956, .9912]]
            configs = [None] * 6 
            for i in range(6):
                configs[i] = deepcopy(config)
                configs[i]['env_config']['env_id'] = i
                configs[i]['env_config']['trial_name'] = configs[i]['env_config']['trial_name'].replace('0', str(i))
            # sus_results = tune.run(algo_trainer,
            #          config=configs[1],
            #          checkpoint_freq=1, name='gridworld',
            #          trial_name_creator=lambda trial: trial_name_string(trial, config),
            #          stop=stops[1],
            #          restore=path_obj['0'])
            env_jumpstart_perfs = np.zeros((len(path_obj), len(path_obj)))
            env_optimal_steps = np.zeros((len(path_obj), len(path_obj)))
            env_agent_steps = np.zeros((len(path_obj), len(path_obj)))
            task_diff_scores = np.zeros((len(path_obj), len(path_obj)))
            test_sim_seed = 123967763
            test_env_seed = 897612344
            config.update({
                'num_workers': 1
            })
            for i, path in path_obj.items():
                i = int(i)
                agent = algo_trainer(config=config, env='gridworld')
                agent.restore(path)
                import code; code.interact(local=vars())
                for j, _ in path_obj.items():
                    j = int(j)
                    print(f'\nTesting agent {i} on environment {j}...')
                    env = envs[j].copy()
                    env.random_state = np.random.RandomState(test_env_seed)
                    # TODO: change n to reasonable number: 1000? change num_sims?
                    perf, optimal_steps, agent_steps = test_env(env, agent, n=1000, num_sims=100, lstm=lstm, sim_random_state=np.random.RandomState(test_sim_seed))
                    print(perf, optimal_steps, agent_steps)
                    # i, j means agent i in environment j
                    env_jumpstart_perfs[i, j] = perf
                    env_optimal_steps[i, j] = optimal_steps
                    env_agent_steps[i, j] = agent_steps
                    task_diff_score = envs[i].graph.compare2(envs[j].graph)
                    print('Task diff score:', task_diff_score)
                    task_diff_scores[i, j] = task_diff_score
            task_sim_scores = 1 - task_diff_scores
            print(task_sim_scores)
            print(env_jumpstart_perfs)
            corr_matrix = np.corrcoef(task_sim_scores.flatten(), env_jumpstart_perfs.flatten())
            print(corr_matrix)
            import code; code.interact(local=vars())
        else:
            agent = algo_trainer(config=config, env='gridworld')
            restore_path = args.test if not last else last_path
            print('Restoring from', restore_path)
            agent.restore(args.test if not last else last_path)

            print('Testing restored agent..')
            import code; code.interact(local=vars())
            perf = test_env(envs[env_id], agent, n=100)
            print(perf)

