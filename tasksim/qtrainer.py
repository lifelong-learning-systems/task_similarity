from curriculum_tools import curriculum
from numpy.core.fromnumeric import var
from numpy.lib.function_base import average
from numpy.lib.npyio import save
import tasksim.gridworld_generator as gen
from tasksim.train_environment import EnvironmentBuilder, test_env
from tasksim.environment import MDPGraphEnv
import random
import numpy as np
from tasksim.experiments.pipeline_utilities import progress_bar
import tasksim.structural_similarity as sim
import matplotlib.pyplot as plt
import time
import json
from os import path
import curriculum_tools
from statistics import mean
import seaborn as sns
import argparse
from collections import deque

#Reference https://medium.com/swlh/introduction-to-q-learning-with-openai-gym-2d794da10f3d
class QTrainer:
    
    def __init__(self, env: MDPGraphEnv, agent_path=None, save=True, override_hyperparameters = False, learning=True, lr=0.01, gamma=0.95, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0, decay=1e-6):
        """
        QTrainer trains q agent 

        :param agent_path: path to store agent metadata and q table (json format)
        :param save: save agent after training
        :param override_hyperparameters: change hyperparameters of existing agent
        :learning: enable epsilon greedy choice and update q table 
        """
        
        self.env = env
        self.num_states = env.graph.P.shape[1]
        
        self.Q = None #Q table
        self.learning = learning 
        self.agent_path = agent_path

        if agent_path == None:
            save = False
        self.save = save  

        json_exists = False if agent_path is None else path.exists(agent_path)

        if json_exists:
            self._load_table(agent_path)    
        else:
            self._create_table()
            override_hyperparameters = True

        if override_hyperparameters:
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.min_epsilon = min_epsilon
            self.max_epsilon = max_epsilon
            self.decay = decay
            self.episode = 0
            self.rewards = []
            self.steps = []
        
        self.total_rewards = 0

    def _load_table(self, path):

        with open(path, 'r') as file:
            
            obj = json.loads(file.read())
            
            self.lr = obj['lr']
            self.gamma = obj['gamma']
            self.epsilon = obj['epsilon']
            self.min_epsilon = obj['min_epsilon']
            self.max_epsilon = obj['max_epsilon']
            self.decay = obj['decay']
            self.Q = np.asarray_chkfinite(obj['Q'])
            self.rewards = obj['rewards']
            self.steps = obj['steps']
            self.episode = obj['episode']

            assert len(self.Q) == self.num_states, 'error: Qtable statespace mismatch'
            

    def _create_table(self):
        self.Q = np.zeros((self.num_states, 4))
        
    # Return new instance of QTrainer with transfered weights
    # pass in S, A as distances/metrics
    def transfer_to(self, other, direct_transfer=True, S=None, A=None):
        
        if direct_transfer:
            other.Q = np.copy(self.Q)
            return
        
        if A is None or S is None:
            S, A, _, _ = self.env.graph.compare(other.env.graph)
        S = 1 - S
        A = 1 - A
        S_total = S.sum().sum()
        A_total = A.sum().sum()

        if not direct_transfer:
            S = S / S_total
            A = A / A_total
        new_states, new_actions = other.Q.shape
        other.Q = np.zeros(other.Q.shape)
        old_states, old_actions = self.Q.shape
        for s_i in range(new_states):
            for a_i in range(new_actions):
                for s_j in range(old_states):
                    for a_j in range(old_actions):
                        w_s = S[s_i, s_j]
                        w_a = A[a_i, a_j]
                        # TODO: determine transfer methodology: linear combine? State transfer only? State transfer argmax?
                        if not direct_transfer:
                            w = 0.5*w_s + 0.5*w_a
                        else:
                            # TODO: actually do 4.2 State Transfer from the paper
                            import pdb; pdb.set_trace()
                            w = 1 if S[s_i, :].argmax() == s_j else 0
                        other.Q[s_i, a_i] += w*self.Q[s_j, a_j]

    def reset(self, center=False):
        self.episode += 1
        return self.env.reset(center=False)

    #step using Q table or provided action
    def step(self, action=None, center=False):
        
        if(action == None):
            action = self._choose_action()

        old_grid_state = self.env.state
        old_mdp_state = self.env.grid_to_state(old_grid_state)

        graph_action = self.env.graph.out_s[old_mdp_state][action]
        transitions = self.env.graph.P[graph_action]
        indices = np.array([i for i in range(len(transitions))])
        new_mdp_state = self.env.random_state.choice(indices, p=transitions)
        new_grid_state = self.env.state_to_grid(new_mdp_state)
        self.env.state = new_grid_state
        reward = self.env.graph.R[graph_action][new_mdp_state]

        if self.learning:
            self.Q[old_mdp_state, action] = self.Q[old_mdp_state, action]+self.lr*(reward+self.gamma*np.max(self.Q[new_mdp_state, :])-self.Q[old_mdp_state, action])
            self.epsilon = max(self.min_epsilon, self.epsilon - self.decay)
            
        row, col = self.env.row_col(new_grid_state)
        done = self.env.grid[row, col] == 2
            
        return self.env.gen_obs(center=center), reward, done, {}

    #choose an acation based on epsilon greedy choice    
    def _choose_action(self):
        
        prob_random = random.uniform(0, 1)

        if not self.learning or prob_random > self.epsilon:
            grid_state = self.env.state
            mdp_state = self.env.grid_to_state(grid_state)
            action = np.argmax(self.Q[mdp_state, :])
        else:
            action = self.env.action_space.sample()
        
        return action

    def compute_action(self, _):
        return self._choose_action()

    def run(self, num_iters=7500, episodic=True, early_stopping=False, threshold=.8, threshold_N=20, record=False, max_eps=None):
        
        # TODO:
        # if record: (RECORD THE OBSERVATIONS!)
        total_episode = 0
        total_step = 0
        performance = 0
        episodes = list(range(num_iters))
        steps = list(range(num_iters))
        
        tmp_optimal = self.compute_optimal_path(self.env.fixed_start) if early_stopping and self.env.fixed_start is not None else None
        if tmp_optimal is not None:
            optimal = tmp_optimal[0]
        else:
            optimal = None

        if episodic:    
            for _ in progress_bar(episodes, prefix='Q_training(episodic)', suffix='Complete'):
            
                self.total_rewards = 0
                obs = self.env.reset()
                done = False
                step = 0
                
                
                while not done and step < 1e6:
                    obs, reward, done, _ = self.step()
                    self.total_rewards += reward
                    step += 1
                    total_step += 1
                
                total_episode += 1

                if optimal is not None: 
                    performance = optimal / step
                    if performance >= threshold:
                        if self.save:
                            self.save_table()
                        return total_episode, total_step, performance
                    

                self.rewards.append(self.total_rewards)
                self.steps.append(step)

        elif not episodic:
            self.total_rewards = 0
            obs = self.env.reset()
            done = False
            step = 0
            for _ in progress_bar(steps, prefix='Q_training(step-wise)', suffix='Complete'):
                if done:
                    obs = self.env.reset()
                    done = False
                    self.rewards.append(self.total_rewards)
                    self.steps.append(step)
                    

                    step = 0
                    total_episode += 1
                    if total_episode >= threshold_N:
                        avg_perf = np.array(self.steps[-threshold_N:]).mean()
                        if optimal is not None:
                            if total_episode % 1000 == 0:
                                print(f'\tAvg perf percent: {optimal/avg_perf}')
                            performance = optimal / avg_perf
                            if performance >= threshold:
                                if self.save:
                                    self.save_table()
                                return total_episode, total_step, performance
                        if max_eps is not None and total_episode >= max_eps:
                            if self.save:
                                self.save_table()
                            return total_episode, total_step, performance
                
                obs, reward, done, _ = self.step()
                self.total_rewards += reward
                step += 1
                total_step += 1

        if self.save:
            self.save_table()
        
        return total_episode, total_step, performance 


    def save_table(self):

        with open(self.agent_path, 'w') as outfile:
            data = {}
            data['lr'] = self.lr
            data['gamma'] = self.gamma
            data['epsilon'] = self.epsilon
            data['min_epsilon'] = self.min_epsilon
            data['max_epsilon'] = self.max_epsilon
            data['decay'] = self.decay
            data['Q'] = self.Q.tolist()
            data['rewards'] = self.rewards
            data['steps'] = self.steps
            data['episode'] = self.episode
            json.dump(data, outfile)

    #evaluate performace
    def eval(self):

        self.learning = False
        return test_env(self.env, self)

    def compute_optimal_path(self, grid_state):
        graph = self.env.graph
        grid = graph.grid
        width = grid.shape[1]
        strat = graph.strat
        q = deque() # q.append(x) -> pushes on to the right side, q.popleft() -> removes and returns from left

        q.append((grid_state, []))
        found = None
        visited = set()
        # TODO: not actually finding the most optimal path...
        while len(q):
            cur_grid_state, cur_path = q.popleft()
            if cur_grid_state in visited:
                continue
            visited.add(cur_grid_state)
            # left, right, up, down, noop
            # These are all grid states
            adjacent = gen.MDPGraph.get_valid_adjacent(cur_grid_state, grid, strat)
            filtered = [a for a in adjacent if a is not None]

            cur_mdp_state = self.env.grid_to_state(cur_grid_state)
            out_actions = graph.out_s[cur_mdp_state]
            assert len(out_actions) == len(filtered), 'darn, something went wrong'

            for id, f in enumerate(filtered):
                if f == cur_grid_state:
                    continue
                looped = False
                for path_state, _ in cur_path:
                    if path_state == f:
                        looped = True
                        break
                if looped:
                    continue
                action = out_actions[id]
                row = f // width
                col = f - width*row
                next_path = [c for c in cur_path]
                next_path.append((cur_grid_state, action))
                if grid[row, col] == 2:
                    found = (f, next_path)
                    break
                else:
                    q.append((f, next_path))
            if found is not None:
                break
        if found is None:
            return None, None
        _, goal_path = found
        # goal_path is an optimal deterministic path from start_state to a goal_state
        # graph.P: Actions - Distribution among States
        return len(goal_path), goal_path


       
def create_grids(config=None, goal_change=True):
    
    def ravel_base(row, col, rows=7, cols=7):
        return row*cols + col

    def unravel_base(idx, rows=7, cols=7):
        return (idx // cols, idx % cols)

    ENV_SHAPE = (7, 7)
    ravel = lambda x, y: ravel_base(x, y, *ENV_SHAPE)
    unravel = lambda x: unravel_base(x, *ENV_SHAPE)

    base_env = EnvironmentBuilder(ENV_SHAPE) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.0) \
            .set_goals([ravel(0, 6)]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()
        
    envs = []

    if goal_change:
        for goal_loc in range(base_env.grid.shape[0] * base_env.grid.shape[1]):
            curr_env = EnvironmentBuilder(ENV_SHAPE) \
                    .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
                    .set_obstacles(obstacle_prob=0.0) \
                    .set_goals([goal_loc]) \
                    .set_success_prob(1.0) \
                    .set_fixed_start(ravel(6, 6)) \
                    .build()
            envs.append(curr_env)

    else:
        for obstacle_loc in range(base_env.grid.shape[0] * base_env.grid.shape[1]):
            
            curr_env = EnvironmentBuilder(ENV_SHAPE) \
                    .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
                    .set_obstacles([ravel((unravel(obstacle_loc)[0] - 1) % ENV_SHAPE[0], (unravel(obstacle_loc)[1] - 1) % ENV_SHAPE[1]), obstacle_loc, ravel((unravel(obstacle_loc)[0] + 1) % ENV_SHAPE[0], (unravel(obstacle_loc)[1] + 1) % ENV_SHAPE[1])]) \
                    .set_goals([ravel(0, 6)]) \
                    .set_success_prob(1.0) \
                    .set_fixed_start(ravel(6, 0)) \
                    .build()
            envs.append(curr_env)

    return base_env, envs

def print_graphs(performance_list, similarity_scores, outfile):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,10))

    performance_list_res = performance_list.reshape(base_env.grid.shape[0], base_env.grid.shape[1])
    tasksim_list_res = similarity_scores.reshape(base_env.grid.shape[0], base_env.grid.shape[1])
    sns.heatmap(performance_list_res, square=True, ax=ax)
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('performance_' + outfile + '.png')

    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(tasksim_list_res, square=True, ax=ax)
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('tasksim_' + outfile + '.png')

    plt.show()

def save_data(perf, task, file_name):
    with open('data_' + file_name + '.json', 'w') as outfile:
            data = {}
            data['performance'] = perf.tolist()
            data['tasksim'] = task.tolist()
            json.dump(data, outfile)

def ravel(row, col, rows=7, cols=7):
    return (row % rows)*cols + (col % cols)

def unravel(idx, rows=7, cols=7):
    return (idx // cols, idx % cols)

def load_config(config_file_path):
    pass

#TODO based on arguments passed in, create log files
#TODO check out tensorboard, might help
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_path', type=str, default=None, help='path to save baseline agent')
    parser.add_argument('--save_agent', choices=['true', 'false', 't', 'f', 'True', 'False'], default='true', help='whether to save baseline agent or not')
    parser.add_argument('--config_file', type=str, default=None, help='json filepath for generating specific training configurations')
    parser.add_argument('--transfer_type', type=str, choices=['direct', 'inderect', 'copy', 'none'], default='copy', help='type of transfer method')
    parser.add_argument('--num_iters_base', type=int, default=5000, help='number of steps/episodes for base agent (depends on episodic_base)')
    parser.add_argument('--num_iters_transfer', type=int, default=200, help='number of steps/episodes for transfer agent (depends on episodic_transfer)')
    parser.add_argument('--episodic_base', choices=['true', 'false', 't', 'f', 'True', 'False'], default='true', help='whether to run baseline agent training iterations in episodes or timesteps (true for episodes, false for timesteps)')
    parser.add_argument('--episodic_transfer', choices=['true', 'false', 't', 'f', 'True', 'False'], default='false', help='whether to run transfer agent iterations in episodes or timesteps (true for episodes, false for timesteps)')
    parser.add_argument('--outfile', type=str, required=True, help='body of output file for colormap images and performance data (do not include filetype suffix)')
    parser.add_argument('--num_trials', type=int, default=5, help='number of transfer training trials')
    parser.add_argument('--performance_metric', choices=['performance', 'steps_to_optimal'], default='performance', help='whether to measure performance (optimal/steps) or number of steps to get to optimal')
    parser.add_argument('--strategy', choices=['goal_change', 'obstacle_change'], default='goal_change', help='for counterexamples, moving diagnonal obstacles, or for regular, changing goals')
    
    args = parser.parse_args()
    agent_path = args.agent_path
    save_agent = True if args.save_agent in ['true', 'True', 't'] else False
    config_file_name = args.config_file
    direct_transfer = True if args.transfer_type in ['direct', 'copy'] else False
    do_transfer = False if args.transfer_type == 'none' else True
    num_iters_base = args.num_iters_base
    num_iters_transfer = args.num_iters_transfer if args.performance_metric == 'performance' else 500
    episodic_base = True if args.episodic_base in ['true', 'True', 't'] else False
    episodic_transfer = True if args.episodic_transfer in ['true', 'True', 't']  or args.performance_metric == 'steps_to_optimal' else False
    outfile = args.outfile
    config = load_config(args.config_file)
    num_trials = max(1, args.num_trials)
    performance_metric = args.performance_metric
    goal_change = True if args.strategy == 'goal_change' else False

    early_stopping = performance_metric == 'steps_to_optimal'

    base_env, envs = create_grids(config, goal_change=goal_change)
    base_trainer = QTrainer(base_env, agent_path=agent_path, save=save_agent)
    base_trainer.run(num_iters_base, episodic=episodic_base)

    similarity_scores = []
    performance_list = []
    envID = 0

    for env in envs:

        env_coord=(envID // env.grid.shape[0], envID % env.grid.shape[1])
        print("------------------Env {}, coord {}------------------".format(envID, env_coord))
        envID+=1

        S, A, num_iters, _ = env.graph.compare(base_env.graph)
        score = 1 - sim.final_score(S)
        similarity_scores.append(score)
        performance_runs = []

        if env.fixed_start is not None and env.grid[unravel(env.fixed_start)] == 1:
            performance_list.append(0)
            continue
        elif env_coord == (env.fixed_start // env.grid.shape[0], env.fixed_start % env.grid.shape[1]):
            performance_list.append(1.0)
            continue
        
        for _ in range(num_trials):

            trainer = QTrainer(env, decay=1e-5)

            if do_transfer:
                base_trainer.transfer_to(trainer, direct_transfer=direct_transfer)

            total_episode, total_step, performance = trainer.run(num_iters=num_iters_transfer, episodic=episodic_transfer, early_stopping=early_stopping)

            if early_stopping:
                if total_step == num_iters_transfer * 5000:
                    total_step = 100000
                performance_runs.append(total_step)
            else:
                trainer_test = trainer.eval()
                performance = trainer_test[0]
                performance_runs.append(performance)

        performance_avg = mean(performance_runs)
        performance_list.append(performance_avg)
    
    

    performance_list = np.array((performance_list))
    similarity_scores = np.array((similarity_scores))
    
    print_graphs(performance_list, similarity_scores, outfile)
    save_data(performance_list, similarity_scores, outfile)
    #import code; code.interact(local=vars())
