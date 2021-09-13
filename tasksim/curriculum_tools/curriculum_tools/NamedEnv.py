
import gym
import copy


class NamedEnv(gym.Wrapper):

    def __init__(self, env, name):
        super().__init__(env)
        self._name = name

    @property
    def name(self):
        return copy.deepcopy(self._name)

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
