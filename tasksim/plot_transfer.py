# Copyright 2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import argparse
import ast
import glob
import pickle
import sys

import dill
import numpy as np
from matplotlib import pyplot as plt

import tasksim.structural_similarity as sim
from tasksim.qtrainer import QTrainer

N_CHUNKS = 40
PERF_ITER = 2500
DPI = 300
DIFF = False
USE_HAUS = False


def get_completed(steps, measure_iters, key=None):
    if key is None:
        return np.searchsorted(np.cumsum(steps), measure_iters)
    if key in get_completed.cache:
        summed = get_completed.cache[key]
    else:
        summed = np.cumsum(steps)
        get_completed.cache[key] = summed
    return np.searchsorted(summed, measure_iters)


get_completed.cache = {}


def get_completed_list(steps, boundaries, key=None):
    if key is None:
        summed = np.cumsum(steps)
    elif key in get_completed.cache:
        summed = get_completed.cache[key]
    else:
        summed = np.cumsum(steps)
        get_completed.cache[key] = summed

    completed = np.zeros(len(boundaries))
    num_completed = 0
    cur_idx = 0
    for b_idx, b in enumerate(boundaries):
        list_end = False
        while True:
            if cur_idx >= len(summed):
                completed[b_idx] = num_completed
                list_end = True
                break
            if summed[cur_idx] <= b:
                num_completed += 1
                cur_idx += 1
            else:
                completed[b_idx] = num_completed
                break
        if list_end:
            break
        if b_idx + 1 == len(boundaries):
            completed[b_idx] = num_completed
    return completed


def get_all_completed(raw_steps, measure_iters):
    ret = {}
    for metric, data in raw_steps.items():
        completed = []
        for _, steps_list in data.items():
            num_completed = np.mean([get_completed(steps, measure_iters) for steps in steps_list])
            completed.append(num_completed)
        ret[metric] = np.array(completed)
    return ret


def get_performance_curves(raw_steps, measure_iters, chunks=N_CHUNKS):
    get_completed.cache = {}
    boundaries = np.linspace(0, measure_iters, chunks)

    ret = {}
    avgs = {}
    for metric, data in raw_steps.items():
        completed = {}
        # for each source env
        for idx, steps_list in data.items():

            completed_eps = np.zeros((len(steps_list), len(boundaries)))
            for row, steps in enumerate(steps_list):
                completed_eps[row] = get_completed_list(steps, boundaries)
            performance = np.mean(completed_eps, axis=0)
            if DIFF:
                performance = np.diff(performance, prepend=0)

            completed[idx] = performance
        ret[metric] = completed
        total = None
        for idx, perf in completed.items():
            if total is None:
                total = [[x] for x in perf]
            else:
                for x, stored in zip(perf, total):
                    stored.append(x)
        total = [np.array(stored) for stored in total]
        avgs[metric] = [x.mean() for x in total]
    return ret, avgs


def process():
    global DIFF

    file_names = glob.glob(f'{RESULTS_DIR}/*.txt')
    results = {}
    optimals = {}
    raw_steps = {}
    meta = None
    COMPUTE_SCORES = False
    for f in file_names:
        key = f.split('_res.txt')[0].split('/')[-1].title()
        with open(f) as ptr:
            lines = ptr.readlines()
        data_file = f.replace('res.txt', 'data.pkl')
        with open(data_file, 'rb') as ptr:
            data = pickle.load(ptr)

        last_token = f.split('/')[-1]
        envs_file = f.replace(last_token, 'all_envs.dill')
        with open(envs_file, 'rb') as ptr:
            all_envs = dill.load(ptr)
        target_env, source_envs = all_envs['target'], all_envs['sources']
        trainer = QTrainer(target_env, save=False)
        optimal_len, _ = trainer.compute_optimal_path(target_env.fixed_start)

        # make into distance
        scores = []
        haus_scores = []
        optimal_lens = []
        for idx, D in enumerate(data['dist_mats']):
            if key == 'New':
                D = 1 - D
            elif key == 'Uniform':
                D = np.ones(D.shape) / np.ones(D.shape).size
            ns, nt = D.shape
            states1 = np.arange(ns)
            states2 = np.arange(nt)
            optimal_lens.append(optimal_len)
            if USE_HAUS:
                haus = max(sim.directed_hausdorff_numpy(D, states1, states2),
                           sim.directed_hausdorff_numpy(D.T, states2, states1))
                scores.append(haus)
            else:
                dist_score = sim.final_score_song(D)
                scores.append(dist_score)
            haus = max(sim.directed_hausdorff_numpy(D, states1, states2),
                       sim.directed_hausdorff_numpy(D.T, states2, states1))
            haus_scores.append(haus)

        def process_line(line):
            line = line.rstrip()
            line = line[1:-1]
            if ',' in line:
                tokens = line.split(', ')
                return np.array([float(token) for token in tokens])
            return np.array([])

        scores = np.array(scores)
        haus_scores = np.array(haus_scores)
        dists = scores
        haus_dists = haus_scores
        optimal = np.array(optimal_lens)
        optimals[key] = optimal

        tokens = lines[0].split(' ')
        num_source, n_trials = int(tokens[0]), int(tokens[1])
        raw_steps[key] = {}
        for i in range(num_source):
            raw_steps[key][i] = []
            for k in range(n_trials):
                raw_steps[key][i].append(process_line(lines[1 + i * n_trials + k]))

        if key == 'new_dist_normalize'.title():
            continue
        results[key] = (dists, haus_dists, None)
        if meta is None:
            meta = ast.literal_eval(lines[-1])

    reward = meta['reward']
    dim = meta['dim']
    rotate = meta['rotate'] if 'rotate' in meta else False
    meta['rotate'] = rotate
    Y_HEIGHT = 150
    transfer_method = meta['transfer'].title() if 'transfer' in meta else 'Weight_Action'

    all_completed = get_all_completed(raw_steps, 1500)
    for key in raw_steps.keys():
        dists, haus_dists, eps = results[key]
        results[key] = dists, haus_dists, all_completed[key]

    plt.clf()

    DIFF = False
    Y_HEIGHT = Y_HEIGHT / (N_CHUNKS if DIFF else 1)
    all_perfs, avg_perfs = get_performance_curves(raw_steps, PERF_ITER, N_CHUNKS)
    N_SOURCES = None
    n_sources_list = []
    reached_eps = {}
    for metric, all_perf in all_perfs.items():
        N_SOURCES = len(all_perf)
        n_sources_list.append(N_SOURCES)
        last_cnt = np.zeros(N_SOURCES)
        for idx, perf in all_perf.items():
            last_cnt[idx] = perf[-1]
        reached_eps[metric] = last_cnt
    assert np.unique(np.array(n_sources_list)).size == 1, 'Mismatch of number of sources used!'
    meta['n_sources'] = N_SOURCES
    for metric, avg_perf in avg_perfs.items():
        x = np.linspace(0, PERF_ITER, N_CHUNKS)
        y = avg_perf
        plt.plot(x, y, marker='.', label=metric)
    plt.ylabel('Cumulative episodes')
    plt.xlabel('Total Iterations')
    plt.ylim([0, Y_HEIGHT])
    plt.title(f'Performance: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    plt.legend()
    plt.savefig(f'{OUT}/performance.png', dpi=DPI)

    plt.clf()

    score_dict = {key: res[0] for key, res in results.items()}
    haus_score_dict = {key: res[1] for key, res in results.items()}
    return all_perfs, avg_perfs, score_dict, haus_score_dict, optimals, meta


if __name__ == '__main__':
    global RESULTS_DIR
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Which directory to read from',
                        default='final_results/dim9_reward100_num10_weight')
    parser.add_argument('--parent', help='Which nested result dir to read from', default=None)
    args = parser.parse_args()
    parent_dir = args.parent
    if parent_dir is None:
        RESULTS_DIR = args.results
        OUT = RESULTS_DIR
        process()
        sys.exit(0)

    sub_dirs = glob.glob(f'{parent_dir}/dim*')
    full_results = {'weight': {}, 'state': {}}
    full_results_all = {'weight': {}, 'state': {}}
    full_scores = {'weight': {}, 'state': {}}
    full_haus_scores = {'weight': {}, 'state': {}}
    optimal_lengths = {'weight': {}, 'state': {}}
    for dir in sub_dirs:
        RESULTS_DIR = dir
        OUT = dir
        all_perfs, avg_perfs, scores, haus_scores, optimals, meta = process()
        transfer = meta['transfer']
        dim = meta['dim']
        reward = meta['reward']
        rotate = meta['rotate']
        N_SOURCES = meta['n_sources']
        main_key = 'weight' if 'weight' in transfer else 'state'
        if 'action' not in transfer:
            full_results[main_key]['Song'] = avg_perfs['Song']
            full_results[main_key]['Uniform'] = avg_perfs['Uniform']
            full_results[main_key]['New'] = avg_perfs['New']
            full_results_all[main_key]['Song'] = all_perfs['Song']
            full_results_all[main_key]['Uniform'] = all_perfs['Uniform']
            full_results_all[main_key]['New'] = all_perfs['New']
            full_scores[main_key]['Song'] = scores['Song']
            full_scores[main_key]['Uniform'] = scores['Uniform']
            full_scores[main_key]['New'] = scores['New']
            full_haus_scores[main_key]['Song'] = haus_scores['Song']
            full_haus_scores[main_key]['Uniform'] = haus_scores['Uniform']
            full_haus_scores[main_key]['New'] = haus_scores['New']
            optimal_lengths[main_key]['Song'] = optimals['Song']
            optimal_lengths[main_key]['Uniform'] = optimals['Uniform']
            optimal_lengths[main_key]['New'] = optimals['New']
        else:
            full_results[main_key]['New_Action'] = avg_perfs['New']
            full_results_all[main_key]['New_Action'] = all_perfs['New']
            full_scores[main_key]['New_Action'] = scores['New']
            full_haus_scores[main_key]['New_Action'] = haus_scores['New']
            optimal_lengths[main_key]['New_Action'] = optimals['New']

    OUT = parent_dir
    Y_HEIGHT = 150
    rot_str = ' + Rotations' if rotate else ''
    for main_key, avg_perfs in full_results.items():
        plt.clf()
        metrics = [metric for metric in avg_perfs.keys()]
        metrics.sort()
        for metric in metrics:
            avg_perf = avg_perfs[metric]
            x = np.linspace(0, PERF_ITER, N_CHUNKS)
            y = avg_perf
            plt.plot(x, y, marker='.', label=metric)
        plt.ylabel('Cumulative episodes')
        plt.xlabel('Total Iterations')
        plt.ylim([0, Y_HEIGHT])
        plt.title(f'Performance: {main_key.title()} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}{rot_str}')
        plt.legend()
        plt.savefig(f'{OUT}/{main_key}_performance.png', dpi=DPI)
        final_perfs_avg = {key: perf[-1] for key, perf in avg_perfs.items()}
        all_perfs = full_results_all[main_key]
        final_perfs = {key: {idx: perf[-1] for idx, perf in perf_dict.items()} for key, perf_dict in all_perfs.items()}
        scores = full_scores[main_key]
        haus_scores = full_haus_scores[main_key]

        final_data = {}
        final_data['meta'] = meta
        final_data['all_perfs'] = all_perfs
        final_data['avg_perfs'] = avg_perfs
        final_data['final_perfs'] = final_perfs
        final_data['avg_final_perfs'] = final_perfs_avg
        final_data['scores'] = scores
        final_data['haus_scores'] = haus_scores
        final_data['optimals'] = optimal_lengths

        final_out = f'{OUT}/{main_key}_final.dill'
        with open(final_out, 'wb') as f:
            dill.dump(final_data, f)
    plt.clf()
