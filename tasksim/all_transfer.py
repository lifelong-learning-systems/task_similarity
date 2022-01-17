import subprocess
import os
import sys
import argparse
import shutil
import glob

from tasksim.transfer_exp1 import *
from tasksim.create_bases import *

NESTED_DIR = None

def setup_dirs(exist_ok=False):
    def dir_exists(path):
        return os.path.exists(path) and os.path.isdir(path)

    if not dir_exists(BASE_DIR):
        print(f'Creating base dir: {BASE_DIR}')
        os.makedirs(BASE_DIR, exist_ok=exist_ok)
    elif not exist_ok:
        print('Base dir already exists...')
        sys.exit(1)
    

    commands = []
    plot_commands = []
    # args = (directory, dim, rot_str, reward, algo)
    base_cmd = f'python transfer_runner1.py --results %s' \
               f' --dim %d%s --reward %d --num {N}'
    # args = directory
    plot_base_cmd = f'python plot_transfer.py --parent %s'
    for dim in DIMS:
        for rot in ROTS:
            for reward in REWARDS:
                rot_dir_str = 'rot_' if rot else ''
                rot_com_str = ' --rotate' if rot else ''
                sub_dir = f'{NESTED_DIR}/{rot_dir_str}dim{int(dim)}_reward{int(reward)}'
                os.makedirs(sub_dir, exist_ok=exist_ok)
                cmd = base_cmd % (sub_dir, dim, rot_com_str, reward)
                commands.append(cmd)
                plot_cmd = plot_base_cmd % sub_dir
                plot_commands.append(plot_cmd)

    return commands, plot_commands

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Nested base', default='nested_results')
    parser.add_argument('--restore', help='Append to existing as much as possible', action='store_true')
    args = parser.parse_args()

    exist_ok = args.restore
    NESTED_DIR = args.results

    commands, plot_commands  = setup_dirs(exist_ok=exist_ok)
    print('Running transfer experiments synchronously...')
    for idx, cmd in enumerate(commands):
        print('Running', idx, '/', len(commands))
        print('\t' + cmd)
        cmd_split = shlex.split(cmd)
        subprocess.call(cmd_split, start_new_session=True)
    print('Plotting results asynchronously...')
    waiting = []
    for idx, cmd in enumerate(plot_commands):
        print('Running', idx, '/', len(plot_commands))
        print('\t' + cmd)
        cmd_split = shlex.split(cmd)
        p = subprocess.Popen(cmd_split, start_new_session=True)
        waiting.append(p)
    for p in waiting:
        p.wait()
    print('Done!')
