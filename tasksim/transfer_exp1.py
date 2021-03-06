# Copyright 2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import argparse
import os
import pickle
import shlex
import subprocess
import sys
from typing import List

import dill

from tasksim.qtrainer import *
from tasksim.train_environment import EnvironmentBuilder
import tasksim.structural_similarity as sim

OUT_VERSION = 2

ARG_DICT = None
STRAT = gen.ActionStrategy.NOOP_EFFECT_COMPRESS
RESULTS_DIR = 'results_transfer'

# Hyper parameters for qtrainer
GAMMA = 0.1
TEST_ITER = int(1e4)

ALGO_CHOICES = ['both', 'new', 'song', 'uniform']
NEW_ALGOS = ['new']

TRANSFER_METHODS = ['weight', 'weight_action',
                    'state', 'state_action']

def init_algo(metric):
    if metric == 'new':
        sim.REWARD_STRATEGY = sim.RewardStrategy.NORMALIZE_INDEPENDENT
        sim.COMPUTE_DISTANCE = False
    elif metric == 'new_dist':
        sim.REWARD_STRATEGY = sim.RewardStrategy.NOOP
        sim.COMPUTE_DISTANCE = True
    elif metric == 'new_dist_normalize':
        sim.REWARD_STRATEGY = sim.RewardStrategy.NORMALIZE_INDEPENDENT
        sim.COMPUTE_DISTANCE = True
    elif metric == 'song' or metric == 'uniform' or metric == 'rand' or metric == 'empty':
        # No special options 
        pass


# From https://stackoverflow.com/a/54628145
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def create_envs(rotate, num_mazes, dim, prob, prng, obs_max, reward=1):
    dimensions = (dim, dim)
    target_env = None
    source_envs = []

    upper_left = 0
    upper_right = dim - 1
    lower_right = np.prod(dimensions) - 1
    lower_left = dim * (dim - 1)

    # Generate 1 more, since it'll be the target env
    print('Computing grids...')
    rejected = 0
    for i in range(num_mazes + 1):
        print(i, '/', num_mazes + 1, '...')
        while True:
            # Always upper left to bottom right, unless "rotate" set to true
            if rotate:
                orient = prng.rand()
                if orient < 0.25:
                    start = upper_left
                    goal = lower_right
                elif orient < 0.5:
                    start = upper_right
                    goal = lower_left
                elif orient < 0.75:
                    start = lower_right
                    goal = upper_left
                else:
                    start = lower_left
                    goal = upper_right
            else:
                start = 0
                goal = np.prod(dimensions) - 1
            obs_prob = (1.0 + prng.rand() * obs_max) / 2
            obstacles = []
            for state in range(np.prod(dimensions)):
                if state == start or state == goal:
                    continue
                if prng.rand() < obs_prob:
                    obstacles.append(state)
            env = EnvironmentBuilder(dimensions).set_strat(STRAT) \
                .set_goals([goal]) \
                .set_fixed_start(start) \
                .set_success_prob(prob) \
                .set_obs_size(dim) \
                .set_do_render(False) \
                .set_obstacles(obstacles) \
                .build()
            trainer = QTrainer(env, save=False)
            path_len, _ = trainer.compute_optimal_path(start)
            if path_len is not None:
                env.graph.R *= reward
                break
            else:
                rejected += 1
        if i == 0:
            target_env = env
        else:
            source_envs.append(env)
    print('Rejected', rejected, 'grids.')
    return target_env, source_envs


def test_env(target_env, new_Q, label, metric, max_eps=None, restore=False):
    agent_path = f'{RESULTS_DIR}/target_{label}.json'
    if path.exists(agent_path) and not restore:
        print(f'Deleting {agent_path} so as to not restore')
        os.remove(agent_path)
    elif path.exists(agent_path) and restore:
        print(f'Restoring from {agent_path}')

    num_iters = int(TEST_ITER)
    if metric == 'empty':
        trainer = QTrainer(target_env, save=False, lr=GAMMA, min_epsilon=0.1, decay=1e-6)
    else:
        trainer = QTrainer(target_env, save=False, lr=GAMMA, epsilon=0.1, min_epsilon=0.1, decay=0)
    trainer.Q = new_Q
    trainer.run(num_iters, episodic=False, max_eps=max_eps)
    return trainer


def weight_transfer(target_env: MDPGraphEnv, source_envs: List, sim_mats: List, dist_mats: List, source_Qs: List,
                    action_sims: List, metric, transfer_method='weight'):
    assert len(sim_mats), 'Sources must be non empty'

    if 'action' in transfer_method:
        use_action = True
    else:
        use_action = False

    new_states = target_env.graph.P.shape[1]
    n_actions = 4
    new_Q = np.zeros((new_states, 4))
    if metric == 'empty':
        return new_Q

    N = len(sim_mats)
    w_base = 1 / N
    for source_env, sim_mat, dist_mat, source_Q, action_sim in zip(source_envs, sim_mats, dist_mats, source_Qs,
                                                                   action_sims):
        assert (sim_mat == sim.sim_matrix_song(dist_mat)).all() if metric not in NEW_ALGOS else \
            (sim_mat == sim.sim_matrix(dist_mat)).all(), 'Sim & dist mismatch'
        # Setup s.t. sim_mat is distances
        if 'state' in transfer_method:
            # Want to do state transfer:
            sim_mat = dist_mat if metric not in NEW_ALGOS else sim.coallesce_sim(dist_mat)
        other_states, _ = source_Q.shape
        # Randomize sim_mat
        if metric == 'rand':
            tmp = np.random.rand(*sim_mat.shape)
            tmp = tmp / tmp.sum()
            sim_mat = tmp

        # Uniform init
        if metric == 'uniform':
            tmp = np.ones(sim_mat.shape)
            tmp = tmp / tmp.sum()
            sim_mat = tmp

        # Now, make sim_mat have one entry per column, at the lowest distance
        if 'state' in transfer_method:
            tmp = np.zeros(sim_mat.shape)
            # Manually verified: for uniform, it's all 0, for song, it's mostly 0, some 2, and then last one is 98
            # new/new_dist actually look correct
            min_dist_sources = np.argmin(sim_mat, axis=0)
            for target_state in range(new_states):
                tmp[min_dist_sources[target_state], target_state] = 1
            sim_mat = tmp

        assert sim_mat.shape == (other_states, new_states), 'Incorrects sim shape'
        column_sums = np.sum(sim_mat, axis=0)
        for target_state in range(new_states):
            for source_state in range(other_states):
                w_sim = sim_mat[source_state, target_state]
                w_col = column_sums[target_state]
                w = w_base * w_sim / w_col

                if action_sim is None or not use_action:
                    for target_action in range(n_actions):
                        new_Q[target_state, target_action] += w * source_Q[source_state, target_action]
                    continue

                source_actions = source_env.graph.out_s[source_state]
                target_actions = target_env.graph.out_s[target_state]
                action_subset = action_sim[source_actions].T[target_actions].T

                action_subset_distances = sim.coallesce_sim(action_subset)

                least_distant_sources = np.min(action_subset_distances, axis=0)
                least_distant_source_actions = np.argmin(action_subset_distances, axis=0)
                for target_action in range(n_actions):
                    source_action = target_action
                    least_distant_source = least_distant_sources[target_action]
                    if action_subset_distances[target_action, target_action] != least_distant_source:
                        source_action = least_distant_source_actions[target_action]
                    new_Q[target_state, target_action] += w * source_Q[source_state, source_action]

    return new_Q


def perform_exp(metric, dim, prob, num_mazes, rotate, seed, obs_max, reward, transfer_method, restore=False,
                notransfer=False):
    init_algo(metric)

    prng = np.random.RandomState(seed)
    print(f'called with {metric}, {dim}, {prob}, {num_mazes}, {seed}')
    all_envs_path = f'{RESULTS_DIR}/all_envs.dill'
    restored = False
    if path.exists(all_envs_path) and not restore:
        print(f'Deleting {all_envs_path} so as to not restore')
        os.remove(all_envs_path)
    elif path.exists(all_envs_path) and restore:
        print(f'Restoring from {all_envs_path}')
        restored = True

    if not restored:
        target_env, source_envs = create_envs(rotate, num_mazes, dim, prob, prng, obs_max, reward)
        all_envs = {'target': target_env, 'sources': source_envs}
        with open(all_envs_path, 'wb') as f:
            dill.dump(all_envs, f)
    else:
        with open(all_envs_path, 'rb') as f:
            all_envs = dill.load(f)
            target_env, source_envs = all_envs['target'], all_envs['sources']

    num_iters = int(1e7)
    min_eps = 100
    threshold = 0.9
    print('Training agents...')
    trainers = []
    optimal = []
    for i, env in enumerate(source_envs):
        print(f'Training agent {i} / {len(source_envs)}...')
        agent_path = f'{RESULTS_DIR}/source_{i}.json'
        restored = False
        if path.exists(agent_path) and not restore:
            print(f'Deleting {agent_path} so as to not restore')
            os.remove(agent_path)
        elif path.exists(agent_path) and restore:
            print(f'Restoring from {agent_path}')
            restored = True
        trainer = QTrainer(env, agent_path, lr=GAMMA, min_epsilon=0.1, decay=1e-6)
        optimal_len, _ = trainer.compute_optimal_path(env.fixed_start)
        if restored:
            trainers.append(trainer)
            optimal.append(optimal_len)
            continue

        trainer.run(num_iters=num_iters, episodic=False, early_stopping=True, threshold=threshold)
        num_eps = len(trainer.steps)
        if num_eps < min_eps:
            print(f'Training failure, {num_eps} episodes completed. Continuing with more training...')
            assert False
        print(f'Final epsilon {trainer.epsilon}')
        trainers.append(trainer)
        optimal.append(optimal_len)
    for idx, trainer in enumerate(trainers):
        avg_steps = moving_average(trainer.steps, int(0.05 * len(trainer.steps)))
        print(f'agent {idx}, number of steps: {avg_steps[-1]}; optimal: {optimal[idx]}')

    data_path = f'{RESULTS_DIR}/{metric}_data.pkl'
    restored = False
    if path.exists(data_path) and not restore:
        print(f'Deleting {data_path} so as to not restore')
        os.remove(data_path)
    elif path.exists(data_path) and restore:
        print(f'Restoring from {data_path}')
        restored = True
    if not restored:
        scores = []
        dist_mats = []
        sim_mats = []
        action_sims = []
        iters = []
        print('Caching sim/dist matrices...')
        for idx, env in enumerate(source_envs):
            print(f'Comparing MDP {idx} / {len(source_envs)}...')
            if metric in NEW_ALGOS:
                assert sim.COMPUTE_DISTANCE == False, 'Unsupported currently'
                D, A, num_iters, _ = env.graph.compare(target_env.graph)
                score = sim.final_score(D)
                sim_mat = sim.sim_matrix(D)
            else:
                D, num_iters = env.graph.compare_song(target_env.graph)
                A = None
                score = sim.final_score_song(D)
                sim_mat = sim.sim_matrix_song(D)
            print(f'\tNum iters: {num_iters}, score: {score}')
            scores.append(score)
            dist_mats.append(D)
            iters.append(num_iters)
            action_sims.append(A)
            sim_mats.append(sim_mat)
        data = {'scores': scores, 'dist_mats': dist_mats, 'sim_mats': sim_mats, 'action_sims': action_sims}
        print('Pickling data to restore later...')
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

    if notransfer:
        print('Exiting early, without doing transfer!')
        sys.exit(0)

    # Now, do the actual weight transfer
    n_trials = 5 if metric == 'empty' else 50
    first_50_total = None
    total_step = {}
    # End trial early if reaching this many completed episodes...
    max_eps = 201
    for trial in range(n_trials):
        print(f'{metric}, {transfer_method} TRANSFER TRIAL', trial, '/', n_trials)
        idx = 0
        optimal_percents = []
        transferred_trainers = []
        for sim_mat, dist_mat, score, trainer in zip(data['sim_mats'], data['dist_mats'], data['scores'], trainers):
            source_Q = trainer.Q
            label = f'{metric}_{idx}'
            if metric in NEW_ALGOS:
                action_sim = data['action_sims'][idx]
            else:
                action_sim = None
            new_Q = weight_transfer(target_env, [trainer.env], [sim_mat], [dist_mat], [source_Q], [action_sim], metric,
                                    transfer_method=transfer_method)
            new_trainer = test_env(target_env, new_Q, label, metric, max_eps=max_eps, restore=restore)
            num_eps = 20
            optimal_len = optimal[idx]
            if not len(new_trainer.steps):
                percent_optimal = 0
            elif len(new_trainer.steps) < num_eps:
                percent_optimal = optimal_len / np.array(new_trainer.steps).mean()
            else:
                avg_steps = moving_average(new_trainer.steps, 20)
                percent_optimal = optimal_len / avg_steps[-1]
            optimal_percents.append(percent_optimal)
            transferred_trainers.append(new_trainer)
            idx += 1

        if first_50_total is None:
            first_50_total = []
            for idx, x in enumerate(transferred_trainers):
                total_step[idx] = [np.array(x.steps).copy()]
        else:
            for idx, x in enumerate(transferred_trainers):
                total_step[idx].append(np.array(x.steps).copy())
    with open(f'{RESULTS_DIR}/{metric}_res.txt', 'w+') as f:
        f.write(f'{len(transferred_trainers)} {n_trials}\n')
        for i in range(len(transferred_trainers)):
            for k in range(n_trials):
                f.write('[' + ', '.join(['%.5f' % x for x in total_step[i][k]]) + ']\n')
        tmp = ARG_DICT.copy()
        tmp['n_trials'] = n_trials
        tmp['out_version'] = OUT_VERSION
        f.write(str(tmp) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default=ALGO_CHOICES[0], choices=ALGO_CHOICES, help='Which metric to use.')
    parser.add_argument('--results', help='Result directory', default='results_transfer')
    parser.add_argument('--transfer', help='Which transfer method to use', choices=TRANSFER_METHODS,
                        default=TRANSFER_METHODS[0])
    parser.add_argument('--seed', help='Specifies seed for the RNG', default=3257823)
    parser.add_argument('--rotate', help='If true, randomly orient the start/goal locations', action='store_true')
    parser.add_argument('--dim', help='Side length of mazes, for RNG', default=13)
    parser.add_argument('--num', help='Number of source mazes to randomly generate', default=16)
    parser.add_argument('--prob', help='Transition probability', default=1)
    parser.add_argument('--restore', help='Restore or not', action='store_true')
    parser.add_argument('--obsmax', help='Max obs probability, to be averaged with 1.0', default=0.5)
    parser.add_argument('--reward', help='Goal reward', default=1)
    parser.add_argument('--notransfer', help='Exit early, before performing transfer experiment', action='store_true')

    args = parser.parse_args()

    seed = int(args.seed)
    dim = int(args.dim)
    num_mazes = int(args.num)
    prob = float(args.prob)
    prob = max(prob, 0)
    prob = min(prob, 1)
    metric = args.metric
    restore = args.restore
    rotate = args.rotate
    obs_max = float(args.obsmax)
    reward = float(args.reward)
    results = args.results
    transfer_method = args.transfer
    notransfer = args.notransfer
    RESULTS_DIR = results

    ARG_DICT = vars(args)

    bound = lambda metric: perform_exp(metric, dim, prob, num_mazes, rotate, seed, obs_max, reward,
                                       transfer_method, restore=restore, notransfer=notransfer)
    if metric == 'both':
        waiting = []
        for metric in ALGO_CHOICES:
            if metric == 'both':
                continue
            launch_str = ' '.join([str(x) for x in sys.argv])
            metric_str = '--metric both'
            if metric_str in launch_str:
                launch_str = launch_str.replace(metric_str, '')
            launch_str = launch_str + f' --metric {metric}'
            cmd = f'python {launch_str}'
            cmds = shlex.split(cmd)
            p = subprocess.Popen(cmds, start_new_session=True)
            waiting.append(p)
        for p in waiting:
            p.wait()
    else:
        bound(metric)
