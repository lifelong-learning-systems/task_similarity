from numpy.core.fromnumeric import var
from numpy.lib.function_base import average
from numpy.lib.npyio import save
import tasksim.gridworld_generator as gen
from tasksim.environment import MDPGraphEnv
import random
import numpy as np
import tasksim.structural_similarity as sim
import matplotlib.pyplot as plt
import time
import json
from os import path
from statistics import mean
import seaborn as sns
import argparse
from collections import deque
from tqdm import tqdm

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
            for _ in tqdm(episodes, desc='Q_training(episodic)'):
            
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
            for _ in steps:
                obs, reward, done, _ = self.step()
                self.total_rewards += reward
                step += 1
                total_step += 1

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

    def compute_optimal_path(self, grid_state):
        graph = self.env.graph
        grid = graph.grid
        width = grid.shape[1]
        strat = graph.strat
        q = deque() 

        q.append((grid_state, []))
        found = None
        visited = set()
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

