# Copyright 2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.
# Distributed under the terms of the BSD 3-Clause License.

import shutil

from tasksim.transfer_exp1 import *

BASE_DIR = None


def setup_dirs(dim, reward, num, rotate, agents='agent_bases'):
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
        rot_str = 'rot_' if rotate else ''
        source_dir = f'{agents}/{rot_str}dim{int(dim)}_reward{int(reward)}'
        for i in range(num):
            agent = f'source_{i}.json'
            shutil.copy(f'{source_dir}/{agent}', f'{sub_dir}/{agent}')
        for algo in filtered_algos:
            data = f'{algo}_data.pkl'
            shutil.copy(f'{source_dir}/{data}', f'{sub_dir}/{data}')
        all_envs = 'all_envs.dill'
        shutil.copy(f'{source_dir}/{all_envs}', f'{sub_dir}/{all_envs}')
        sub_dirs[method] = sub_dir
    return sub_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Base for result directories', default='runner_results')
    parser.add_argument('--rotate', help='If true, randomly orient the start/goal locations', action='store_true')
    parser.add_argument('--dim', help='Side length of mazes, for RNG', default=9)
    parser.add_argument('--num', help='Number of source mazes to randomly generate', default=10)
    parser.add_argument('--reward', help='Goal reward', default=1)

    args = parser.parse_args()

    dim = int(args.dim)
    num_mazes = int(args.num)
    assert 1 <= num_mazes <= 100, 'Num mazes invalid'
    reward = float(args.reward)
    results = args.results
    rotate = args.rotate
    BASE_DIR = results

    arg_dict = vars(args)

    sub_dirs = setup_dirs(dim, reward, num_mazes, rotate)
    waiting = []
    for method, out_dir in sub_dirs.items():
        rot_str = ' --rotate' if rotate else ''
        cmd = f'python transfer_exp1.py --restore {rot_str} --results {out_dir}' \
              f' --dim {dim} --reward {reward} --num {num_mazes}' \
              f' --metric both --transfer {method}'
        print(cmd)
        cmds = shlex.split(cmd)
        p = subprocess.Popen(cmds, start_new_session=True)
        waiting.append(p)
    for p in waiting:
        p.wait()
