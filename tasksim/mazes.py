from typing import List, Tuple
from tasksim.gridworld_generator import ActionStrategy
from tasksim.qtrainer import *
from tasksim.scenarios import sim_score, dist_score

import argparse


UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


STRAT = ActionStrategy.NOOP_EFFECT_COMPRESS


def create_mazes(dimensions: Tuple[int], paths: List[List[int]]):
    envs = []
    for path in paths:
        env = build_maze(dimensions, path)
        envs.append(env)
    return envs


def build_maze(dimensions: Tuple[int], path: List[int]):
    max_obs = max(dimensions[0], dimensions[1])
    fixed_start = ravel(dimensions[0] - 1, 0, rows=dimensions[0], cols=dimensions[1])
    goal_loc = dimensions[1] - 1
    builder = EnvironmentBuilder(dimensions) \
        .set_strat(STRAT)\
        .set_fixed_start(fixed_start=fixed_start)\
        .set_success_prob(success_prob=1.0)\
        .set_obs_size(max_obs)\
        .set_goals(goal_locations=[goal_loc])

    #generate list of raveled obstacles except starting and goal states
    full_obstacles = [obs for obs in range(dimensions[0] * dimensions[1])]
    full_obstacles.remove(goal_loc)
    full_obstacles.remove(fixed_start)

    pos = fixed_start
    remove_obstacles = []
    for step in path:
        if pos == goal_loc:
            remove_obstacles.pop()
            break
        
        posRow = unravel(pos, rows=dimensions[0], cols=dimensions[1])[0]
        posCol = unravel(pos, rows=dimensions[0], cols=dimensions[1])[1]

        if step == UP:
            pos = max(pos - dimensions[0], 0)
        elif step == DOWN:
            pos = ravel(min(posRow + 1, dimensions[0] - 1), posCol, rows=dimensions[0], cols=dimensions[1])
        elif step == LEFT:
            pos = ravel(posRow, max(posCol - 1, 0), rows=dimensions[0], cols=dimensions[1])
        elif step == RIGHT:      
            pos = ravel(posRow, min(posCol + 1, dimensions[1] - 1), rows=dimensions[0], cols=dimensions[1])

        remove_obstacles.append(pos)
    
    obstacles = [obs for obs in full_obstacles if obs not in remove_obstacles]

    builder = builder.set_obstacles(obstacle_locations=obstacles)

    env = builder.build()
    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    metric_choices = ['new', 'song', 'both']
    parser.add_argument('--metric', default=metric_choices[0], choices=metric_choices, help='Which metric to use.')
    parser.add_argument('--rand', action='store_true', help='Use randomly generated mazes, as per seed')
    parser.add_argument('--seed', help='Specifies seed for the RNG', default=81923673)
    parser.add_argument('--dim', help='Side length of mazes, for RNG', default=9)
    parser.add_argument('--num', help='Number of mazes to randomly generate', default=7)
    parser.add_argument('--prob', help='Transition probability', default=1)
    args = parser.parse_args()

    use_rand = int(args.rand)
    seed = int(args.seed)
    dim = int(args.dim)
    num_mazes = int(args.num)
    prob = float(args.prob)
    prob = max(prob, 0)
    prob = min(prob, 1)
    metric = args.metric

    def perform_exp():
        if not use_rand:
            dimensions = (9, 9)
            empty_env = EnvironmentBuilder(dimensions).set_strat(STRAT).set_goals([ravel(0, 8, *dimensions)]).set_fixed_start(ravel(8, 0, *dimensions)).set_success_prob(1.0).set_obs_size(9).build()

            maze1 = [UP] * 9 + [RIGHT] * 9
            maze2 = [UP, UP, RIGHT, RIGHT] * 4
            maze3 = [RIGHT]*2 + [UP]*2 + [LEFT]*2 + [UP]*2 + [RIGHT]*2 + [UP]*2 + [LEFT]*2 + [UP]*2 + [RIGHT]*4 + [DOWN]*2 + [RIGHT]*2 + [DOWN]*2 + [LEFT]*2 + [DOWN]*2 + [RIGHT]*2 + [DOWN]*2 + [RIGHT]*2 + [UP]*9
            maze4 = [RIGHT]*8 + [UP]*2 + [LEFT]*8 + [UP]*2 + [RIGHT]*8 + [UP]*2 + [LEFT]*8 + [UP]*2 + [RIGHT]*8
            maze5 = [UP]*8 + [RIGHT]*2 + [DOWN]*8 + [RIGHT]*2 + [UP]*8 + [RIGHT]*2 + [DOWN]*8 + [RIGHT]*2 + [UP]*8
            maze6 = [RIGHT]*8 + [UP]*4 + [LEFT]*8 + [UP]*4 + [RIGHT]*8
            

            mazes = [maze1, maze2, maze3, maze4, maze5, maze6]
            envs = create_mazes(dimensions, mazes)
            envs.insert(0, empty_env)
        else:
            dimensions = (dim, dim)
            prng = np.random.RandomState(seed)
            grids = []
            obs_max = 0.4
            envs = []
            # upper right
            goal = ravel(0, 8, *dimensions)
            # bottom left
            start = ravel(8, 0, *dimensions)
            for _ in range(num_mazes):
                obs_prob = prng.rand()*obs_max
                num_states = np.prod(dimensions)
                states = np.arange(num_states)
                states = states[(states != goal) & (states != start) & (states != start + 1) & (states != start - dim) \
                                & (states != goal - 1) & (states != goal + dim)]
                obstacles = []
                for s in states:
                    if prng.rand() < obs_prob:
                        obstacles.append(s)
                env = EnvironmentBuilder(dimensions).set_strat(STRAT).set_goals([goal]) \
                                                        .set_fixed_start(start) \
                                                        .set_success_prob(prob).set_obs_size(9) \
                                                        .set_obstacles(obstacles).build()
                envs.append(env)
        
        sim_scores : List[List[float]] = []
        unique_scores = set()
        precision = 6
        precision_str = f'%.{precision}f'
        for i in range(len(envs)):
            scores = []
            for j in range(len(envs)):
                print(i, j, '...')
                score, S = dist_score(envs[i], envs[j], metric=metric, detailed=True)
                score = float(precision_str % score)
                unique_scores.add(score)
                scores.append(score)

            sim_scores.append(scores)

        cnt = len(envs)
        fig, axs = plt.subplots(cnt, cnt * 2, figsize=(25, 13))

        for i in range(len(axs)):
            num = 0
            idx = 0 
            for j in range(len(axs[i])):
                if num % 2 == 0:
                    #show next idx env
                    axs[i, j].imshow(envs[idx].reset(center=False))

                    title = "score:" +  str(sim_scores[idx][i])
                    axs[i, j].xaxis.set_label_coords(1.25, 1.20)
                    axs[i, j].set_xlabel(title)
                    #increment idx
                    idx += 1
                else:
                    #show current i env
                    axs[i, j].imshow(envs[i].reset(center=False))
                
                num += 1
                

        fig.tight_layout(pad=0.2)

        rand_str = '_rand' if use_rand else ''
        prob_str = f'_{prob}' if prob != 1 else ''
        plt.savefig(f'maze_out/maze_{metric}{rand_str}{prob_str}.png')
        print(sim_scores)
        print(unique_scores)
    if metric == 'both':
        metric = 'new'
        perform_exp()
        metric = 'song'
        perform_exp()
    else:
        perform_exp()
    
    
    


    