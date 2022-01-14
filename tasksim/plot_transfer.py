import numpy as np
from matplotlib import pyplot as plt
import glob
import ast
import argparse
import tasksim
import tasksim.structural_similarity as sim
import pickle

N_CHUNKS = 100
DIFF = False
USE_HAUS = False


def get_completed(steps, measure_iters, min=0):
    tmp = np.cumsum(steps) - min
    tmp = np.array(tmp)
    tmp = tmp[tmp >= 0]
    return np.searchsorted(tmp, measure_iters)

def get_all_completed(raw_steps, measure_iters):
    ret = {}
    for metric, data in raw_steps.items():
        completed = []
        for _, steps in data.items():
            num_completed = get_completed(steps, measure_iters)
            completed.append(num_completed)
        ret[metric] = np.array(completed)
    return ret

def get_performance_curves(raw_steps, measure_iters, chunks=N_CHUNKS):

    boundaries = np.linspace(0, measure_iters, chunks)

    ret = {}
    avgs = {}
    for metric, data in raw_steps.items():
        completed = {}
        # for each source env
        for idx, steps in data.items():
            performance = []
            for boundary_idx, boundary in enumerate(boundaries):
                if boundary_idx == 0:
                    performance.append(0)
                    continue
                prev_completed = get_completed(steps, boundaries[boundary_idx-1])
                num_completed = get_completed(steps, boundary)
                if DIFF:
                    performance.append(num_completed - prev_completed)
                else:
                    performance.append(num_completed)
            completed[idx] = performance
        ret[metric] = completed
        total = None
        for idx, perf in completed.items():
            # if total is None:
            #     total = np.array(perf).copy()
            # else:
            #     total += np.array(perf).copy()
            if total is None:
                total = [[x] for x in perf]
            else:
                for x, stored in zip(perf, total):
                    stored.append(x)
        total = [np.array(stored) for stored in total]
        avgs[metric] = [x.mean() for x in total]
        #avgs[metric] = [np.median(x) for x in total]
    return ret, avgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Which directory to read from', default='final_results/dim9_reward100_num10_weight')
    args = parser.parse_args()
    RESULTS_DIR = args.results
    OUT = RESULTS_DIR


    file_names = glob.glob(f'{RESULTS_DIR}/*.txt')
    results = {}
    raw_steps = {}
    meta = None
    for f in file_names:
        key = f.split('_res.txt')[0].split('/')[-1].title()
        with open(f) as ptr:
            lines = ptr.readlines()
        data_file = f.replace('res.txt', 'data.pkl')
        with open(data_file, 'rb') as ptr:
            data = pickle.load(ptr)
        # make into distance
        haus_scores = []
        for D in data['dist_mats']:
            if key == 'New':
                D = sim.coallesce_sim(D)
            elif key == 'Uniform':
                D = np.ones(D.shape) / np.ones(D.shape).size
            ns, nt = D.shape
            states1 = np.arange(ns)
            states2 = np.arange(nt)
            haus = max(sim.directed_hausdorff_numpy(D, states1, states2), sim.directed_hausdorff_numpy(D.T, states2, states1))
            haus_scores.append(haus)
        
        def process_line(line):
            line = line.rstrip()
            line = line[1:-1]
            tokens = line.split(', ')
            return np.array([float(token) for token in tokens])
        iters_per_ep = process_line(lines[0])
        haus_scores = np.array(haus_scores)
        dists = process_line(lines[1])
        if USE_HAUS:
            dists = haus_scores
        eps_in_1000 = process_line(lines[2])


        num_source = int(lines[3])
        raw_steps[key] = {}
        for i in range(4, 4+num_source):
            raw_steps[key][i-4] = process_line(lines[i])

        if key == 'new_dist_normalize'.title():
            continue
        results[key] = (dists, iters_per_ep, eps_in_1000)
        if meta is None:
            meta = ast.literal_eval(lines[-1])

    reward = meta['reward']
    dim = meta['dim']
    Y_HEIGHT = 150 if dim == '9' else 50
    transfer_method = meta['transfer'].title() if 'transfer' in meta else 'Weight_Action'

    all_completed = get_all_completed(raw_steps, 1500)
    for key in raw_steps.keys():
        dists, iters, eps = results[key]
        results[key] = dists, iters, all_completed[key]

    plt.clf()
    PERF_ITER = 2500

    DIFF = False    
    Y_HEIGHT = Y_HEIGHT/(N_CHUNKS if DIFF else 1)
    all_perfs, avg_perfs = get_performance_curves(raw_steps, PERF_ITER, N_CHUNKS)
    N_SOURCES = None
    reached_eps = {}
    for metric, all_perf in all_perfs.items():
        N_SOURCES = len(all_perf)
        last_cnt = np.zeros(N_SOURCES)
        for idx, perf in all_perf.items():
            last_cnt[idx] = perf[-1]
        reached_eps[metric] = last_cnt
    for metric, avg_perf in avg_perfs.items():
        x = np.linspace(0, PERF_ITER, N_CHUNKS)
        y = avg_perf
        plt.plot(x, y, marker='.', label=metric)
    plt.ylabel('Cumulative episodes')
    plt.xlabel('Total Iterations')
    plt.ylim([0, Y_HEIGHT])
    plt.title(f'Performance: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    plt.legend()
    plt.savefig(f'{OUT}/performance.png', dpi=200)

    plt.clf()

    DIFF = True
    Y_HEIGHT = Y_HEIGHT/(N_CHUNKS if DIFF else 1)
    all_perfs, avg_perfs = get_performance_curves(raw_steps, PERF_ITER, N_CHUNKS)
    N_SOURCES = None
    for _, all_perf in all_perfs.items():
        N_SOURCES = len(all_perf)
        break
    for metric, avg_perf in avg_perfs.items():
        x = np.linspace(0, PERF_ITER, N_CHUNKS)
        y = avg_perf
        plt.plot(x, y, marker='.', label=metric)
    plt.ylabel('Delta Episodes')
    plt.xlabel('Total Iterations')
    plt.ylim([0, Y_HEIGHT])
    plt.title(f'Performance Gradient: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    plt.legend()
    plt.savefig(f'{OUT}/performance_grad.png', dpi=200)

    plt.clf()
    baseline_dists, baseline_iters, baseline_eps = results['Uniform']
    epsilon = 1e-6


    print()
    fig, axs = plt.subplots(1, len(results))
    for ax, data in zip(axs, results.items()):
        metric, vals = data
        dists, iters, eps = vals
        reached_ep = reached_eps[metric]
        # R = np.corrcoef(dists, iters)[0, 1]
        # print(f'Metric {metric}: R = {R}')
        # ax.set_title(metric)
        # ax.scatter(dists, iters, label=('R = %.2f' % R))
        # ax.legend()
        R = np.corrcoef(dists, reached_ep)[0, 1]
        print(f'Metric {metric}: R = {R}')
        ax.set_title(metric)
        ax.scatter(dists, reached_ep, label=('R = %.2f' % R))
        ax.legend()

    figure = plt.gcf()
    figure.set_size_inches(8*len(results), 6)
    fig.tight_layout()
    plt.savefig(f'{OUT}/correlation_iters.png')




    # print()
    # for metric, vals in results.items():
    #     dists, iters, eps = vals
    #     idxs = np.arange(len(iters))
    #     speedup = baseline_iters/iters
    #     #speedup = iters
    #     plt.plot(idxs, speedup, marker='.', label=metric)
    #     print(f'Metric {metric}: avg iters speedup: {speedup.mean()}')
    #     print(f'Metric {metric}: median iters speedup: {np.median(speedup)}')
    # plt.ylabel('Speedup')
    # plt.xlabel('Source Index')
    # plt.title(f'Iters vs. Song Speedup: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    # plt.legend()
    # plt.savefig(f'{OUT}/speedup_iters.png', dpi=200)

    # plt.clf()
    

    # print()
    # for metric, vals in results.items():
    #     dists, iters, eps = vals
    #     idxs = np.arange(len(iters))
    #     speedup = eps/baseline_eps
    #     #speedup = iters
    #     plt.plot(idxs, speedup, marker='.', label=metric)
    #     print(f'Metric {metric}: avg episodes speedup: {speedup.mean()}')
    #     print(f'Metric {metric}: median episodes speedup: {np.median(speedup)}')
    # plt.ylabel('Speedup')
    # plt.xlabel('Source Index')
    # plt.title(f'Eps vs. Song Speedup: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    # plt.legend()
    # plt.savefig(f'{OUT}/speedup_eps.png', dpi=200)

    # plt.clf()

    # print()
    # for metric, vals in results.items():
    #     dists, iters, eps = vals
    #     idxs = np.arange(len(iters))
    #     #speedup = iters
    #     plt.plot(idxs, iters, marker='.', label=metric)
    #     print(f'Metric {metric}: avg iter: {iters.mean()}')
    #     print(f'Metric {metric}: median iter: {np.median(iters)}')

    # plt.ylabel('Iterations')
    # plt.xlabel('Source Index')

    # plt.title(f'Avg. Iterations in 50 Episodes: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    # plt.legend()
    # plt.savefig(f'{OUT}/raw_iters.png', dpi=200)

    # plt.clf()

    # print()
    # for metric, vals in results.items():
    #     dists, iters, eps = vals
    #     idxs = np.arange(len(iters))
    #     #speedup = iters
    #     plt.plot(idxs, eps, marker='.', label=metric)
    #     print(f'Metric {metric}: avg eps: {iters.mean()}')
    #     print(f'Metric {metric}: median eps: {np.median(iters)}')

    # plt.ylabel('Iterations')
    # plt.xlabel('Source Index')

    # plt.title(f'Avg. Episodes in 10,000 Iterations: {transfer_method} transfer w/ Reward {reward}, Dim {dim}, N={N_SOURCES}')
    # plt.legend()
    # plt.savefig(f'{OUT}/raw_eps.png', dpi=200)