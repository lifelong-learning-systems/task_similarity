
import gym
import numpy as np

class NoisyEnv(gym.Env):

	def __init__(self, env, noise=0.1):
		self.env = env
		self.noise = noise

	@property
	def observation_space(self):
		return self.env.observation_space

	@property
	def action_space(self):
		return self.env.action_space

	def add_noise(self, state):
		state = np.asarray(state, dtype=np.float32)
		state = state + np.random.uniform(
			low=-128.0*self.noise,
			high=128.0*self.noise, 
			size=state.shape)

		state = np.clip(state, 0.0, 255.0)
		state = state.astype(np.uint8)

		return state

	def reset(self):
		return self.add_noise(self.env.reset())

	def step(self, action):
		state, r, d, info = self.env.step(action)
		return self.add_noise(state), r, d, info

	def render(self, mode=None):
		self.env.render(mode)
