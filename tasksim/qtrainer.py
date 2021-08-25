import tasksim.gridworld_generator as gen
from tasksim.train_environment import EnvironmentBuilder
from tasksim.environment import MDPGraphEnv
import random
import numpy as np
from tasksim.experiments.pipeline_utilities import progress_bar
import matplotlib.pyplot as plt
import time
import json
from os import path

import argparse

class QTrainer():

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
            .set_success_prob(1)\
            .build()
    
    trainer = QTrainer(env, "agent1.json", learning=(not test))
    trainer.run()
    print(trainer.Q)
    plt.ion()
    plt.scatter([i for i in range(len(trainer.rewards))], trainer.rewards, 0.2)
    plt.show()