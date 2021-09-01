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
    def __init__(self, env: MDPGraphEnv, agent_path, override_hyperparameters = True, lr=0.01, gamma=0.95, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0, decay=0.01, learning=True):
        self.env = env
        self.Q = None
        self.agent_path = agent_path
        json_exists = path.exists(agent_path)
        

        if json_exists:
            self._load_table(agent_path)    
        else:
            self._create_table()

        if override_hyperparameters:
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
            self.min_epsilon = min_epsilon
            self.max_epsilon = max_epsilon
            self.decay = decay
            self.learning = learning
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

            assert self.env.graph.grid.shape[0] * self.env.graph.grid.shape[1] == len(self.Q), 'error, '
            

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

        if prob_random > self.epsilon:
            action = np.argmax(self.Q[self.env.state, :])
        else:
            action = self.env.action_space.sample()
        
        return action

    def compute_action(self, _):
        return self._choose_action()

    def run(self, num_episodes = 10000):
        
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
    env = EnvironmentBuilder((7, 7)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.9)\
            .build()
    env2 = EnvironmentBuilder((7, 7)) \
            .set_obstacles(obstacle_prob=0.1, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.7)\
            .build()
    env3 = EnvironmentBuilder((7, 7)) \
            .set_obstacles(obstacle_prob=0.1, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_goals([6*7+6]) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .set_success_prob(0.7)\
            .build()
    
    trainer = QTrainer(env, "agent1.json", learning=(not test))
    trainer.run(1000)

    plt.ion()
    plt.scatter([i for i in range(len(trainer.rewards))], trainer.rewards, 0.2)
    plt.show()
    trainer = QTrainer(env, "agent1.json", learning=False)
    trainer2_baseline = QTrainer(env2, "agent2_baseline.json", learning=True)
    trainer2_transfer = QTrainer(env2, "agent2_transfer.json", learning=True)
    trainer.transfer_to(trainer2_transfer)
    trainer3_baseline = QTrainer(env3, "agent3_baseline.json", learning=True)
    trainer3_transfer = QTrainer(env3, "agent3_transfer.json", learning=True)
    trainer.transfer_to(trainer3_transfer)
    
    #print('Baseline (Agent 1):', test_env(env, trainer))
    num_eps = 200
    trainer2_baseline.run(num_eps)
    trainer2_transfer.run(num_eps)
    trainer3_baseline.run(num_eps)
    trainer3_transfer.run(num_eps)
    print('Agent 2 baseline:', test_env(env2, trainer2_baseline))
    print('Agent 2 transfer:', test_env(env2, trainer2_transfer))
    print('Agent 3 baseline:', test_env(env3, trainer3_baseline))
    print('Agent 3 transfer:', test_env(env3, trainer3_transfer))
    import code; code.interact(local=vars())
