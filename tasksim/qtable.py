import tasksim.gridworld_generator as gen
from tasksim.train_environment import EnvironmentBuilder
from tasksim.environment import MDPGraphEnv
import random
import numpy as np
from tasksim.experiments.pipeline_utilities import progress_bar

class QTrainer():

    def __init__(self, env: MDPGraphEnv, lr=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.01, max_epsilon=1.0, decay=0.01, learning=False):
        self.env = env
        self.Q = []
        self.create_table()
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay = decay
        self.learning = learning
        self.total_rewards = 0
        self.episode = 0

    def create_table(self):
        num_states = self.env.graph.grid.shape[0] * self.env.graph.grid.shape[1]
        self.Q = np.zeros((num_states, 4))
        
    def reset(self):
        self.episode += 1
        return self.env.reset()

    def qstep(self, action=None):

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
            self.epsilon = self.min_epsilon+(self.max_epsilon-self.min_epsilon)*np.exp(-self.decay*self.episode)

        return self.env.gen_obs(), reward, done, {}
    
    def _choose_action(self):
        prob_random = random.uniform(0, 1)

        if prob_random > self.epsilon:
            action = np.argmax(self.Q[self.env.state, :])
        else:
            action = self.env.action_space.sample()
        
        return action

    def run(self, num_episodes = 10000):
        
        episodes = list(range(num_episodes))
        for _ in progress_bar(episodes, prefix='Q_training', suffix='Complete'):
            obs = self.env.reset()
            done = False
            while not done:
                obs, reward, done, _ = self.qstep()
                self.total_rewards += reward

def main():

    base_seed = 41239678
    transition_seed = 94619456
    env = EnvironmentBuilder((7, 7)) \
            .set_obstacles(obstacle_prob=0.2, obstacle_random_state=np.random.RandomState(base_seed)) \
            .set_step_reward(-0.001) \
            .set_obs_size(7) \
            .build()
    
    trainer = QTrainer(env, learning=True)
    trainer.run()
    print(trainer.Q)


    
    


if __name__ == '__main__':
    main()