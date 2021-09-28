from curriculum_tools import curriculum
from numpy.core.fromnumeric import var
from numpy.lib.function_base import average
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

import argparse

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
        
        self.total_rewards = 0

    def _load_table(self, path):

        with open(path, 'r') as file:
            
            obj = json.loads(file.read())
            
            self.lr = obj['lr']
            self.gamma = obj['gamma']
            self.epsilon = obj['epsion']
            self.min_epsilon = obj['min_epsilon']
            self.max_epsilon = obj['max_epsilon']
            self.decay = obj['decay']
            self.Q = np.asarray_chkfinite(obj['Q'])
            self.rewards = obj['rewards']
            self.episode = obj['episode']

            assert self.env.graph.grid.shape[0] * self.env.graph.grid.shape[1] == len(self.Q), 'error: Qtable statespace mismatch'
            

    def _create_table(self):
        num_states = self.env.graph.grid.shape[0] * self.env.graph.grid.shape[1]
        self.Q = np.zeros((num_states, 4))
        
    # Return new instance of QTrainer with transfered weights
    # pass in S, A as distances/metrics
    def transfer_to(self, other, direct_transfer=True, S=None, A=None):
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

    def reset(self):
        self.episode += 1
        return self.env.reset()

    #step using Q table or provided action
    def step(self, action=None):

        if(action == None):
            action = self._choose_action()

        old_state = self.env.state

        graph_action = self.env.graph.out_s[self.env.state][action]
        transitions = self.env.graph.P[graph_action]
        indices = np.array([i for i in range(len(transitions))])
        state = self.env.random_state.choice(indices, p=transitions)
        self.env.state = state
        reward = self.env.graph.R[graph_action][state]

        new_state = self.env.state
        
        if self.learning:
            self.Q[old_state, action] = self.Q[old_state, action]+self.lr*(reward+self.gamma*np.max(self.Q[new_state, :])-self.Q[old_state, action])
            self.epsilon = max(0, self.epsilon - self.decay)
            
        
        row, col = self.env.row_col(state)
        done = self.env.grid[row, col] == 2
            
        return self.env.gen_obs(), reward, done, {}

    #choose an acation based on epsilon greedy choice    
    def _choose_action(self):
        
        prob_random = random.uniform(0, 1)

        if not self.learning or prob_random > self.epsilon:
            action = np.argmax(self.Q[self.env.state, :])
        else:
            action = self.env.action_space.sample()
        
        return action

    def compute_action(self, _):
        return self._choose_action()

    def run(self, num_episodes=5000):
        
        episodes = list(range(num_episodes))
        for _ in progress_bar(episodes, prefix='Q_training', suffix='Complete'):
        
            self.total_rewards = 0
            obs = self.env.reset()
            done = False

            while not done:
                obs, reward, done, _ = self.step()
                self.total_rewards += reward

            self.rewards.append(self.total_rewards)
        
        if self.save == True:
            self.save_table()


    def save_table(self):

        with open(self.agent_path, 'w') as outfile:
            data = {}
            data['lr'] = self.lr
            data['gamma'] = self.gamma
            data['epsion'] = self.epsilon
            data['min_epsilon'] = self.min_epsilon
            data['max_epsilon'] = self.max_epsilon
            data['decay'] = self.decay
            data['Q'] = self.Q.tolist()
            data['rewards'] = self.rewards
            data['episode'] = self.episode
            json.dump(data, outfile)

    #evaluate performace
    def eval(self):

        self.learning = False
        return test_env(self.env, self)

       
def create_grids():
    

    def ravel(row, col, rows=7, cols=7):
        return row*cols + col

    def unravel(idx, rows=7, cols=7):
        return (idx // cols, idx % cols)

    #TODO put goal in top right (on diagonal down-left), remove obstacles, deterministic do "curriculum" with goal in all other possible squares
    #in each grid, put difference in performance over (50?) runs, for goal in each location 
    #create heatmap for each
    #do the same for task similarity

    base_env = EnvironmentBuilder((9, 9)) \
            .set_obstacles(obstacle_prob=0.0) \
            .set_goals([ravel(1, 7, 9, 9)]) \
            .set_success_prob(1.0) \
            .build()
    
    envs = []

    for goal_loc in range(base_env.grid.shape[0] * base_env.grid.shape[1]):
        curr_env = EnvironmentBuilder((base_env.grid.shape)) \
                .set_obstacles(obstacle_prob=0.0) \
                .set_goals([goal_loc]) \
                .set_success_prob(1.0) \
                .build()
        envs.append(curr_env)

    return base_env, envs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='whether to disable learning')
    args = parser.parse_args()
    test = args.test

    base_env, envs = create_grids()
    base_trainer = QTrainer(base_env, agent_path="base_agent.json")
    base_trainer.run(10000)

    similiarity_scores = []
    performance_gains = []
    envID = 1
    for env in envs:
        print("------------------Env {}------------------".format(envID))
        envID += 1
        S, A, num_iters, _ = env.graph.compare(base_env.graph)
        score = 1 - sim.final_score(S)
        similiarity_scores.append(score)

        performance_runs = []
        num_trials = 5
        num_episodes = 200
        for i in range(num_trials):

            trainer = QTrainer(env, decay=1e-4)

            base_trainer.transfer_to(trainer, direct_transfer=False)

            trainer.run(num_episodes=num_episodes)

            trainer_test = trainer.eval()

            performance = trainer_test[0]
            performance_runs.append(performance)

        performance_avg = mean(performance_runs)
        performance_gains.append(performance_avg)
    
    plt.ion()
    plt.scatter(performance_gains, similiarity_scores)
    plt.show()
    import code; code.interact(local=vars())
