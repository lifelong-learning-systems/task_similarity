import subprocess
import os
import sys
import argparse
import shutil
import glob

from tasksim.transfer_exp1 import *

BASE_DIR = None
DIMS = [9, 13]
N = 100
ROTS = [True, False]
REWARDS = [1, 100]

def setup_dirs(exist_ok=False):
    def dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)

    if not dir_exists(BASE_DIR):
        print(f'Creating base dir: {BASE_DIR}')
        os.makedirs(BASE_DIR, exist_ok=exist_ok)
    elif not exist_ok:
        print('Base dir already exists...')
        sys.exit(1)
    

    initial_algo = 'new'
    filtered_algos = [algo for algo in ALGO_CHOICES if algo != 'both' and algo != initial_algo]

    stage1 = []
    stage2 = []
    # args = (directory, dim, rot_str, reward, algo)
    base_cmd = f'python transfer_exp1.py --restore --results %s' \
               f' --dim %d%s --reward %d --num {N}' \
               f' --metric %s --notransfer'
    for dim in DIMS:
        for rot in ROTS:
            for reward in REWARDS:
                rot_dir_str = 'rot_' if rot else ''
                rot_com_str = ' --rotate' if rot else ''
                sub_dir = f'{BASE_DIR}/{rot_dir_str}dim{int(dim)}_reward{int(reward)}'
                os.makedirs(sub_dir, exist_ok=exist_ok)
                initial = base_cmd % (sub_dir, dim, rot_com_str, reward, initial_algo)
                others = [base_cmd % (sub_dir, dim, rot_com_str, reward, algo) for algo in filtered_algos]
                stage1.append(initial)
                stage2.extend(others)

    return stage1, stage2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Base for agent bases', default='agent_bases')
    parser.add_argument('--restore', help='Append to existing as much as possible', action='store_true')
    args = parser.parse_args()

    exist_ok = args.restore
    BASE_DIR = args.results

    stage1, stage2 = setup_dirs(exist_ok=exist_ok)
    waiting1 = []
    print('Launching Stage 1...')
    for cmd in stage1:
        print('\t' + cmd)
        cmd_split = shlex.split(cmd)
        p = subprocess.Popen(cmd_split, start_new_session=True)
        waiting1.append(p)
    for p in waiting1:
        p.wait()
    
    waiting2 = []
    print('Launching Stage 2...')
    for cmd in stage2:
        print('\t' + cmd)
        cmd_split = shlex.split(cmd)
        p = subprocess.Popen(cmd_split, start_new_session=True)
        waiting2.append(p)
    for p in waiting2:
        p.wait()
    print('Done!')