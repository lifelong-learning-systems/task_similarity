import os
from typing import List

from numpy.lib.utils import source
import tasksim
import tasksim.structural_similarity as sim
import tasksim.gridworld_generator as gen

import argparse
import numpy as np
import ot
import pickle

from tasksim.qtrainer import *

ARG_DICT = None
STRAT = gen.ActionStrategy.NOOP_EFFECT_COMPRESS
RESULTS_DIR = 'results_transfer'

ALGO_CHOICES = ['both', 'new', 'song', 'new_dist', 'uniform', 'empty']#, 'new_dist_normalize']
NEW_ALGOS = ['new', 'new_dist', 'new_dist_normalize']

# rand_sim 
TRANSFER_METHODS = ['weight', 'weight_action']

SCORE_METHODS = ['emd', 'haus']

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

# Hyper parameters for qtrainer
GAMMA = 0.1
TEST_ITER = int(1e5)


# From https://stackoverflow.com/a/54628145
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_envs(num_mazes, dim, prob, prng, obs_max, reward=1):
    dimensions = (dim, dim)
    # Always upper left to bottom right
    start = 0
    goal = np.prod(dimensions) - 1
    target_env = None
    source_envs = []
    # Generate 1 more, since it'll be the target env
    # Then, generate 10x the amount, and select the top `num_mazes` (in terms of distance) to be compared to?
    print('Computing grids...')
    rejected = 0
    for i in range(num_mazes + 1):
        print(i, '/', num_mazes+1, '...')
        while True:
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
                #print('No path found, generating new grid!')
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
        restored = True
    
    num_iters = int(TEST_ITER)
    #trainer = QTrainer(target_env, save=False, lr=GAMMA, epsilon=0.2, min_epsilon=0.1, decay=5.0/num_iters)
    if metric == 'empty':
        trainer = QTrainer(target_env, save=False, lr=GAMMA, min_epsilon=0.1, decay=1e-6)
    else:
        trainer = QTrainer(target_env, save=False, lr=GAMMA, epsilon=0.1, min_epsilon=0.1, decay=0)
    trainer.Q = new_Q
    trainer.run(num_iters, episodic=False, max_eps=max_eps)
    return trainer

def weight_transfer(target_env: MDPGraphEnv, source_envs: List, sim_mats: List, source_Qs: List, action_sims: List, metric, transfer_method='weight'):
    assert len(sim_mats), 'Sources must be non empty'

    if transfer_method == 'weight_action':
        use_action = True
    else:
        use_action = False
    
    new_states = target_env.graph.P.shape[1]
    # TODO: don't assume 4 actions, but whatever
    n_actions = 4
    action_dist = [1.0/n_actions for _ in range(n_actions)]
    new_Q = np.zeros((new_states, 4))
    if metric == 'empty':
        return new_Q

    N = len(sim_mats)
    w_base = 1/N
    for source_env, sim_mat, source_Q, action_sim in zip(source_envs, sim_mats, source_Qs, action_sims):

        other_states, _ = source_Q.shape
        # Randomize sim_mat
        if metric == 'rand':
            tmp = np.random.rand(*sim_mat.shape)
            tmp = tmp / tmp.sum()
            sim_mat = tmp

        if metric == 'uniform':
            tmp = np.ones(sim_mat.shape)
            tmp = tmp / tmp.sum()
            sim_mat = tmp

        assert sim_mat.shape == (other_states, new_states), 'Incorrects sim shape'
        column_sums = np.sum(sim_mat, axis=0)
        #action_sim_transpose = action_sim.T
        for target_state in range(new_states):
            for source_state in range(other_states):
                w_sim = sim_mat[source_state, target_state]
                w_col = column_sums[target_state]
                w = w_base*w_sim/w_col

                # TODO: figure out action-space correspondence first!
                col_order = np.arange(n_actions)
                if action_sim is not None and use_action:
                    source_actions = source_env.graph.out_s[source_state]
                    target_actions = target_env.graph.out_s[target_state]
                    action_subset = action_sim[source_actions].T[target_actions].T
                    action_mat = sim.sim_matrix(action_subset.copy())
                    assert action_mat[action_mat != 0].size == n_actions, 'Unexpected number of entries'
                    col_order = np.argmax(action_mat, axis=1)
                
                for target_action in range(n_actions):
                    new_Q[target_state, target_action] += w*source_Q[source_state, col_order[target_action]]
                #new_Q[target_state, :] += w*source_Q[source_state, :]
    return new_Q


def perform_exp(metric, dim, prob, num_mazes, seed, obs_max, reward, transfer_method, score_method, restore=False):
    init_algo(metric)

    prng = np.random.RandomState(seed)
    print(f'called with {metric}, {dim}, {prob}, {num_mazes}, {seed}')
    target_env, source_envs = create_envs(num_mazes, dim, prob, prng, obs_max, reward)

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
        # TODO: Reduce decay if needed
        trainer = QTrainer(env, agent_path, lr=GAMMA, min_epsilon=0.1, decay=1e-6)
        optimal_len, _ = trainer.compute_optimal_path(env.fixed_start)
        if restored:
            trainers.append(trainer)
            optimal.append(optimal_len)
            continue

        trainer.run(num_iters=num_iters, episodic=False, early_stopping=True, threshold=threshold, record=True)
        num_eps = len(trainer.steps)
        if num_eps < min_eps:
            print(f'Training failure, {num_eps} episodes completed. Continuing with more training...')
            assert False
        print(f'Final epsilon {trainer.epsilon}')
        trainers.append(trainer)
        optimal.append(optimal_len)
    for idx, trainer in enumerate(trainers):
        avg_steps = moving_average(trainer.steps, int(0.05*len(trainer.steps)))
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
    
    
    # Now, do the actual weight transfer
    # TODO: measure performance, average many results lol
    n_trials = 10 
    first_50_total = None
    completed_total = None
    total_step = {}
    # End trial early if reaching this many completed episodes...
    measure_eps = 50
    max_eps = measure_eps + 1
    measure_iters = 1e4
    for trial in range(n_trials):
        print('TRANSFER TRIAL', trial, '/', n_trials)
        idx = 0
        optimal_percents = []
        transferred_trainers = []
        for sim_mat, score, trainer in zip(data['sim_mats'], data['scores'], trainers):
            source_Q = trainer.Q
            label = f'{metric}_{idx}'
            if metric in NEW_ALGOS:
                action_sim = data['action_sims'][idx]
            else:
                action_sim = None
            new_Q = weight_transfer(target_env, [trainer.env], [sim_mat], [source_Q], [action_sim], metric, transfer_method=transfer_method)
            print(f'Testing transfer source {idx} via metric {metric} to target for {TEST_ITER} steps...')
            new_trainer = test_env(target_env, new_Q, label, metric, max_eps=max_eps, restore=restore)
            print(f'\tDistance score: {score}')
            print(f'\tEpisodes completed: {len(new_trainer.steps)}')
            num_eps = 20
            optimal_len = optimal[idx]
            if not len(new_trainer.steps):
                percent_optimal = 0
                print(f'\tLast {0} episode avg. steps: {np.inf}, percent of optimal: {percent_optimal}')
            elif len(new_trainer.steps) < 20:
                percent_optimal = optimal_len / np.array(new_trainer.steps).mean()
                print(f'\tLast {len(new_trainer.steps)} episode avg. steps: {np.array(new_trainer.steps).mean()}'\
                    f', percent of optimal: {percent_optimal}')
            else:
                avg_steps = moving_average(new_trainer.steps, 20)
                percent_optimal = optimal_len / avg_steps[-1]
                print(f'\tLast {num_eps} episode avg. steps: {avg_steps[-1]}'\
                    f', percent of optimal: {percent_optimal}')
            optimal_percents.append(percent_optimal)
            transferred_trainers.append(new_trainer)
            idx += 1
        
        def get_completed(steps):
            tmp = np.cumsum(steps)
            return np.searchsorted(tmp, measure_iters)
        completed_eps = np.array([get_completed(x.steps[:measure_eps]) for x in transferred_trainers])
        first_50 = np.array([np.array(x.steps[:measure_eps]).mean() for x in transferred_trainers])
        if first_50_total is None:
            first_50_total = first_50.copy()
            completed_total = completed_eps.copy()
            for idx, x in enumerate(transferred_trainers):
                total_step[idx] = np.array(x.steps).copy()
        else:
            first_50_total += first_50
            completed_total += completed_eps
            for idx, x in enumerate(transferred_trainers):
                total_step[idx] += np.array(x.steps)
    first_50_avg = first_50_total/n_trials
    completed_eps_avg = completed_total/n_trials
    with open(f'{RESULTS_DIR}/{metric}_res.txt', 'w+') as f:
        f.write('[' + ', '.join(['%.5f' % x for x in first_50_avg]) + ']\n')
        f.write('[' + ', '.join([('%.5f' % x) for x in data['scores']]) + ']\n')
        f.write('[' + ', '.join(['%.5f' % x for x in completed_eps_avg]) + ']\n')
        f.write(f'{len(transferred_trainers)}\n')
        for i in range(len(transferred_trainers)):
            f.write('[' + ', '.join(['%.5f' % x for x in total_step[i]/n_trials]) + ']\n')
        tmp = ARG_DICT.copy()
        tmp['n_trials'] = n_trials
        f.write(str(tmp) + '\n')


# Larger reward test: dim 9, prob 1.0, num 8, obsmax 0.5 [R = 100, song only]
# Prep for exp2: dim 9, prob 1.0, num 100, obsmax 0.5

# Initial larger grid: dim 13, prob 1.0, num 16, obsmax 0.4
# Experiment w/o annealing dim 9, prob 1.0, num 8, obsmax 0.5
# Experiment sent on slack: dim 9, prob 1.0, num 8, obsmax 0.5 [annealing starts at 0.2 epsilon, decay 5.0/num_iters]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default=ALGO_CHOICES[0], choices=ALGO_CHOICES, help='Which metric to use.')
    parser.add_argument('--results', help='Result directory', default='results_transfer')
    parser.add_argument('--transfer', help='Which transfer method to use', choices=TRANSFER_METHODS, default=TRANSFER_METHODS[0])
    parser.add_argument('--score', help='Which scoring method to use', choices=SCORE_METHODS, default=SCORE_METHODS[0])
    parser.add_argument('--seed', help='Specifies seed for the RNG', default=3257823)
    parser.add_argument('--dim', help='Side length of mazes, for RNG', default=13)
    parser.add_argument('--num', help='Number of source mazes to randomly generate', default=16)
    parser.add_argument('--prob', help='Transition probability', default=1)
    parser.add_argument('--restore', help='Restore or not', action='store_true')
    parser.add_argument('--obsmax', help='Max obs probability, to be averaged with 1.0', default=0.5)
    parser.add_argument('--reward', help='Goal reward', default=1)
    
    # TODO: cache params passed in, save in output directory; pass in RESULTS_DIR rather than assuming
    args = parser.parse_args()

    seed = int(args.seed)
    dim = int(args.dim)
    num_mazes = int(args.num)
    prob = float(args.prob)
    prob = max(prob, 0)
    prob = min(prob, 1)
    metric = args.metric
    restore = args.restore
    obs_max = float(args.obsmax)
    reward = float(args.reward)
    results = args.results
    transfer_method = args.transfer
    score_method = args.score
    RESULTS_DIR = results

    ARG_DICT = vars(args)

    bound = lambda metric: perform_exp(metric, dim, prob, num_mazes, seed, obs_max, reward, transfer_method, score_method, restore=restore)
    if metric == 'both':
        for metric in ALGO_CHOICES:
            if metric == 'both':
                continue
            bound(metric)
    else:
        bound(metric)
    
    
    


