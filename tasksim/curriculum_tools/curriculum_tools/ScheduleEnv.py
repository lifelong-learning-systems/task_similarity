
# Generic utilities to maintain schedules of environments.
# These are used by meta arcade to create curricula.
# For meta arcade specifically, it is recommended to use the curriculum build tools 
# instead of these classes directly.


import gym
import gym.spaces
import numpy as np

class ScheduleEnv(gym.Env):

    def __init__(self, schedule=[], by_episode=False):

        # a schedule is a list of entries [env, duration]
        self.schedule = schedule
        self.by_episode = by_episode

        self.action_spaces = [e[0].action_space for e in schedule]

        union_actions = np.max([a.n for a in self.action_spaces])
        self.action_space = gym.spaces.Discrete(union_actions)

        self.step_count = 1
        self.episode_count = 1

        #convert schedule counts to cumulative
        summ = 0
        for s in self.schedule:
            summ += s[1]
            s[1] = summ

        self.env = self.schedule[0][0]
        self.env_idx = 0

        self.needs_env_change = False


    # look up the current env based on step count
    def select_env_episodic(self):
        if self.episode_count > self.schedule[self.env_idx][1]:
            if self.env_idx < len(self.schedule)-1:
                self.env_idx += 1
                self.env = self.schedule[self.env_idx][0]

    def select_env_stepwise(self):
        if self.step_count > self.schedule[self.env_idx][1]:
            if self.env_idx < len(self.schedule)-1:
                self.env_idx += 1
                self.env = self.schedule[self.env_idx][0]
                return True
        return False

    def check_env_expired(self):
        if self.by_episode:
            if self.episode_count > self.schedule[self.env_idx][1]:
                if self.env_idx < len(self.schedule)-1:
                    return True
        else:
            if self.step_count > self.schedule[self.env_idx][1]:
                if self.env_idx < len(self.schedule)-1:
                    return True

        return False

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped


    def reset(self):
        if self.needs_env_change:
            if self.by_episode:
                self.select_env_episodic()
            else:
                self.select_env_stepwise()

        return self.env.reset()


    def step(self, action):
        self.step_count += 1
        action_limit = self.action_spaces[self.env_idx].n
        if action >= action_limit:
            s, r, done, info = self.env.step(0)
        else:
            s, r, done, info = self.env.step(action)

        # check if the environment needs re-evaluating
        if not self.by_episode:
            self.needs_env_change = self.check_env_expired()
            if self.needs_env_change:
                done = True
            return s, r, done, info

        else:
            if done:
                self.episode_count += 1
                self.needs_env_change = True
            return s, r, done, info


    def render(self, mode=None):
        self.env.render(mode)


    def seed(self, seed):
        for s in self.schedule:
            s[0].seed(seed)