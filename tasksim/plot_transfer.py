import numpy as np
from matplotlib import pyplot as plt
import glob
import ast
import argparse

OUT='plot_out'

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

def get_performance_curves(raw_steps, measure_iters, chunks=20):

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
                    continue
                prev_completed = get_completed(steps, boundaries[boundary_idx-1])
                num_completed = get_completed(steps, boundary)
                #performance.append(num_completed - prev_completed)
                performance.append(num_completed)
            completed[idx] = performance
        ret[metric] = completed
        total = None
        for idx, perf in completed.items():
            if total is None:
                total = np.array(perf).copy()
            else:
                total += np.array(perf).copy()
        avgs[metric] = total / len(completed)
    return ret, avgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Which directory to read from', default='final_results/dim9_reward100_num10_weight')
    args = parser.parse_args()
    RESULTS_DIR = args.results


    file_names = glob.glob(f'{RESULTS_DIR}/*.txt')
    results = {}
    raw_steps = {}
    meta = None
    for f in file_names:
        key = f.split('_res.txt')[0].split('/')[-1].title()
        with open(f) as ptr:
            lines = ptr.readlines()
        
        def process_line(line):
            line = line.rstrip()
            line = line[1:-1]
            tokens = line.split(', ')
            return np.array([float(token) for token in tokens])
        iters_per_ep = process_line(lines[0])
        dists = process_line(lines[1])
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
    transfer_method = meta['transfer'].title() if 'transfer' in meta else 'Weight_Action'

    all_completed = get_all_completed(raw_steps, 1500)
    for key in raw_steps.keys():
        dists, iters, eps = results[key]
        results[key] = dists, iters, all_completed[key]

    plt.clf()
    PERF_ITER = 2500
    CHUNKS = 20
    _, avg_perfs = get_performance_curves(raw_steps, PERF_ITER, CHUNKS)
    for metric, avg_perf in avg_perfs.items():
        x = np.linspace(0, PERF_ITER, CHUNKS - 1)
        y = avg_perf
        plt.plot(x, y, marker='.', label=metric)
    plt.ylabel('Cumulative episodes')
    plt.xlabel('Total Iterations')
    plt.title(f'Performance: {transfer_method} transfer w/ Reward {reward}, Dim {dim}')
    plt.legend()
    plt.savefig(f'{OUT}/performance.png', dpi=200)

    plt.clf()
    baseline_dists, baseline_iters, baseline_eps = results['Song']
    epsilon = 1e-6
    print()
    for metric, vals in results.items():
        dists, iters, eps = vals
        idxs = np.arange(len(iters))
        speedup = baseline_iters/iters
        #speedup = iters
        plt.plot(idxs, speedup, marker='.', label=metric)
        print(f'Metric {metric}: avg iters speedup: {speedup.mean()}')
        print(f'Metric {metric}: median iters speedup: {np.median(speedup)}')
    plt.ylabel('Speedup')
    plt.xlabel('Source Index')
    plt.title(f'Iters vs. Song Speedup: {transfer_method} transfer w/ Reward {reward}, Dim {dim}')
    plt.legend()
    plt.savefig(f'{OUT}/speedup_iters.png', dpi=200)

    plt.clf()
    

    print()
    for metric, vals in results.items():
        dists, iters, eps = vals
        idxs = np.arange(len(iters))
        speedup = eps/baseline_eps
        #speedup = iters
        plt.plot(idxs, speedup, marker='.', label=metric)
        print(f'Metric {metric}: avg episodes speedup: {speedup.mean()}')
        print(f'Metric {metric}: median episodes speedup: {np.median(speedup)}')
    plt.ylabel('Speedup')
    plt.xlabel('Source Index')
    plt.title(f'Eps vs. Song Speedup: {transfer_method} transfer w/ Reward {reward}, Dim {dim}')
    plt.legend()
    plt.savefig(f'{OUT}/speedup_eps.png', dpi=200)

    plt.clf()

    print()
    for metric, vals in results.items():
        dists, iters, eps = vals
        idxs = np.arange(len(iters))
        #speedup = iters
        plt.plot(idxs, iters, marker='.', label=metric)
        print(f'Metric {metric}: avg iter: {iters.mean()}')
        print(f'Metric {metric}: median iter: {np.median(iters)}')

    plt.ylabel('Iterations')
    plt.xlabel('Source Index')

    plt.title(f'Avg. Iterations in 50 Episodes: {transfer_method} transfer w/ Reward {reward}, Dim {dim}')
    plt.legend()
    plt.savefig(f'{OUT}/raw_iters.png', dpi=200)

    plt.clf()

    print()
    for metric, vals in results.items():
        dists, iters, eps = vals
        idxs = np.arange(len(iters))
        #speedup = iters
        plt.plot(idxs, eps, marker='.', label=metric)
        print(f'Metric {metric}: avg eps: {iters.mean()}')
        print(f'Metric {metric}: median eps: {np.median(iters)}')

    plt.ylabel('Iterations')
    plt.xlabel('Source Index')

    plt.title(f'Avg. Episodes in 10,000 Iterations: {transfer_method} transfer w/ Reward {reward}, Dim {dim}')
    plt.legend()
    plt.savefig(f'{OUT}/raw_eps.png', dpi=200)


    print()
    fig, axs = plt.subplots(1, len(results))
    for ax, data in zip(axs, results.items()):
        metric, vals = data
        dists, iters, eps = vals
        R = np.corrcoef(dists, iters)[0, 1]
        print(f'Metric {metric}: R = {R}')
        ax.set_title(metric)
        ax.scatter(dists, iters, label=('R = %.2f' % R))
        ax.legend()

    #fig.tight_layout()
    #plt.savefig(f'{OUT}/correlation_iters.png')
