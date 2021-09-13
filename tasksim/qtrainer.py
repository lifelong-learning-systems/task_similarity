from numpy.core.fromnumeric import var
import tasksim.gridworld_generator as gen
from tasksim.train_environment import EnvironmentBuilder, test_env
from tasksim.environment import MDPGraphEnv
import random
import numpy as np
from tasksim.experiments.pipeline_utilities import progress_bar
import matplotlib.pyplot as plt
import time
import json
from os import path

import argparse

class QTrainer:
    def __init__(self, env: MDPGraphEnv, agent_path=None, no_save=False, override_hyperparameters = False, lr=0.01, gamma=0.95, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0, decay=0.01, learning=True):
        self.env = env
        self.Q = None
        self.learning = learning
        self.agent_path = agent_path

        if agent_path == None:
            no_save = True
        
        json_exists = path.exists(agent_path)

        self.no_save = no_save        

        if json_exists and not self.no_save:
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


        if not self.learning:
            self.epsilon = 0
        
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
                            w = 1 if S[s_i, :].argmax() == s_j else 0
                        other.Q[s_i, a_i] += w*self.Q[s_j, a_j]

    def reset(self):
        self.episode += 1
        return self.env.reset()

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
        #print("Old State: {}\n New State: {}\n".format(old_state, new_state))

        if self.learning:
            self.Q[old_state, action] = self.Q[old_state, action]+self.lr*(reward+self.gamma*np.max(self.Q[new_state, :])-self.Q[old_state, action])
        
        #print(self.Q)

        row, col = self.env.row_col(state)
        done = self.env.grid[row, col] == 2

        if self.learning:
            self.epsilon = self.epsilon - (self.decay/7000)

        return self.env.gen_obs(), reward, done, {}
        
    def _choose_action(self):
        
        prob_random = random.uniform(0, 1)

        if not self.learning or prob_random > self.epsilon:
            action = np.argmax(self.Q[self.env.state, :])
        else:
            action = self.env.action_space.sample()
        
        return action

    def compute_action(self, _):
        return self._choose_action()

    def run(self, num_episodes = 5000):
        
        episodes = list(range(num_episodes))
        for _ in progress_bar(episodes, prefix='Q_training', suffix='Complete'):
        #for i in range(num_episodes):
            self.total_rewards = 0
            obs = self.env.reset()
            done = False
            while not done:
                obs, reward, done, _ = self.step()
                self.total_rewards += reward
            #print(self.epsilon)
            self.rewards.append(self.total_rewards)
        
        if self.no_save:
            return

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='whether to disable learning')
    args = parser.parse_args()
    test = args.test
    

    base_seed = 41239678
    transition_seed = 94619456
    env = EnvironmentBuilder((9, 9)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.9)\
            .build()
    env2 = EnvironmentBuilder((9, 9)) \
            .set_obstacles(obstacle_prob=0.3, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.65)\
            .set_goals([4*9+6])\
            .build()
    env3 = EnvironmentBuilder((7, 7)) \
            .set_obstacles(obstacle_prob=0.1, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_goals([6*7+6]) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.7)\
            .build()
    
    trainer = QTrainer(env, "agent1.json", learning=(not test))
    trainer.run(10000)
    

    # plt.ion()
    # plt.scatter([i for i in range(len(trainer.rewards))], trainer.rewards, 0.2)
    # plt.show()

    agent2diff_arr = []
    agent3diff_arr = []
    agent2base_test_arr = []
    agent3base_test_arr = []
    agent2trans_test_arr = []
    agent3trans_test_arr = []
    transfer_trials = 100


    for i in range(transfer_trials):

        print("------------------\nCurrent trial: ", i + 1, "\n------------------")
        #trainer = QTrainer(env, "agent1.json", learning=False)
        trainer2_baseline = QTrainer(env2, "agent2_baseline.json", learning=True, no_save=True)
        trainer2_transfer = QTrainer(env2, "agent2_transfer.json", learning=True, no_save=True)
        trainer.transfer_to(trainer2_transfer, direct_transfer=False)

        # trainer3_baseline = QTrainer(env3, "agent3_baseline.json", learning=True, no_save=True)
        # trainer3_transfer = QTrainer(env3, "agent3_transfer.json", learning=True, no_save=True)
        # trainer.transfer_to(trainer3_transfer, direct_transfer=False)
        
        #print('Baseline (Agent 1):', test_env(env, trainer))
        num_eps = 100
        trainer2_baseline.run(num_eps)
        trainer2_transfer.run(num_eps)
        # trainer3_baseline.run(num_eps)
        # trainer3_transfer.run(num_eps)

        trainer2_baseline.learning = False
        trainer2_transfer.learning = False
        # trainer3_baseline.learning = False
        # trainer3_transfer.learning = False
    
        agent2base_test = test_env(env2, trainer2_baseline)
        agent2base_test_arr.append(agent2base_test)

        agent2trans_test = test_env(env2, trainer2_transfer)
        agent2trans_test_arr.append(agent2trans_test)

        agent2diff = agent2trans_test[0] - agent2base_test[0]
        agent2diff_arr.append(agent2diff)

        # agent3base_test = test_env(env3, trainer3_baseline)
        # agent3base_test_arr.append(agent3base_test)

        # agent3trans_test = test_env(env3, trainer3_transfer)
        # agent3trans_test_arr.append(agent3trans_test)

        # agent3diff = agent3trans_test[0] - agent3base_test[0]
        # agent3diff_arr.append(agent3diff)


        # print('Agent 2 baseline:', agent2base_test)
        # print('Agent 2 transfer:', agent2trans_test)
        # print('Agent 3 baseline:', agent3base_test)
        # print('Agent 3 transfer:', agent3trans_test)
        
    
    n_bins = 20
    plt.ion()
    plt.hist(agent2diff_arr, bins=n_bins)
    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # axs[0].hist(agent2diff_arr, bins=n_bins)
    # axs[1].hist(agent3diff_arr, bins=n_bins)
    plt.show()

    import code; code.interact(local=vars())
