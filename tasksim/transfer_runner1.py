import subprocess
import os
import sys
import argparse
import shutil
import glob

from tasksim.transfer_exp1 import *

BASE_DIR = None

def setup_dirs(dim, reward, num, agents='agent_bases'):
    agent_sources = glob.glob(f'{agents}/dim{int(dim)}_reward{int(reward)}/*.json')
    agent_sources.sort(key=lambda x: int(x.split('source_')[-1].split('.')[0]))

    def dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)

    if not dir_exists(BASE_DIR):
        print(f'Creating base dir: {BASE_DIR}')
        os.makedirs(BASE_DIR)
    else:
        print('Base dir already exists...')
        sys.exit(1)
    
    sub_dirs = {}
    filtered_algos = [algo for algo in ALGO_CHOICES if algo != 'both']
    for method in TRANSFER_METHODS:
        sub_dir = f'{BASE_DIR}/dim{int(dim)}_reward{int(reward)}_num{num}_{method}'
        os.makedirs(sub_dir)
        source_dir = f'{agents}/dim{int(dim)}_reward{int(reward)}'
        for i in range(num):
            agent = f'source_{i}.json'
            shutil.copy(f'{source_dir}/{agent}', f'{sub_dir}/{agent}')
        for algo in filtered_algos:
            data = f'{algo}_data.pkl'
            shutil.copy(f'{source_dir}/{data}', f'{sub_dir}/{data}')
        sub_dirs[method] = sub_dir
    return sub_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Base for result directories', default='runner_results')
    #parser.add_argument('--seed', help='Specifies seed for the RNG', default=3257823)
    parser.add_argument('--dim', help='Side length of mazes, for RNG', default=9)
    parser.add_argument('--num', help='Number of source mazes to randomly generate', default=10)
    #parser.add_argument('--prob', help='Transition probability', default=1)
    #parser.add_argument('--restore', help='Restore or not', action='store_true')
    parser.add_argument('--reward', help='Goal reward', default=1)
    
    # TODO: cache params passed in, save in output directory; pass in RESULTS_DIR rather than assuming
    args = parser.parse_args()

    #seed = int(args.seed)
    dim = int(args.dim)
    num_mazes = int(args.num)
    assert 1 <= num_mazes <= 100, 'Num mazes invalid'
    # prob = float(args.prob)
    # prob = max(prob, 0)
    # prob = min(prob, 1)
    #restore = args.restore
    reward = float(args.reward)
    results = args.results
    BASE_DIR = results

    arg_dict = vars(args)

    sub_dirs = setup_dirs(dim, reward, num_mazes)
    waiting = []
    for method, out_dir in sub_dirs.items():
        cmd = f'python transfer_exp1.py --restore --results {out_dir}' \
              f' --dim {dim} --reward {reward} --num {num_mazes}' \
              f' --metric both --transfer {method}'
        print(cmd)
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True)
        waiting.append(p)
    for p in waiting:
        p.wait()
