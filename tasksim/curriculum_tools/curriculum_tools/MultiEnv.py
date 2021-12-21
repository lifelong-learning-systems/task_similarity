import gym
import numpy as np


# gym environment which draws from a bag of other environments each episode
# TODO: base this selection on a fixed seed so it is repeatable

# Notes:
# Only works with discrete action spaces, taking the union of all action spaces from provided envs
# If an action does not apply to the current environment, 0 will be passed in its place (no-op)

class MultiEnv(gym.Env):

    def __init__(self, env_list=[], p=None):

        self.envs = env_list
        self.probabilities = [1.0/float(len(env_list)) for e in env_list] if (p is None) else p

        self.observation_spaces = [e.observation_space for e in self.envs]
        self.action_spaces = [e.action_space for e in self.envs]

        union_actions = np.max([a.n for a in self.action_spaces])
        self.action_space = gym.spaces.Discrete(union_actions)

        self.select_env()


    def select_env(self):
        idxs = list(range(len(self.envs)))
        self.selected_idx = np.random.choice(idxs, p=self.probabilities)
        self.env = self.envs[self.selected_idx]


    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped


    def reset(self):
        self.select_env()
        return self.env.reset()


    def step(self, action):
        action_limit = self.action_spaces[self.selected_idx].n
        if action >= action_limit:
            return self.env.step(0)
        else:
            return self.env.step(action)


    def render(self, mode=None):
        self.env.render(mode)
