import gym
import curriculum_tools
import cv2
import numpy as np
from noisy_env_helper import NoisyEnv

# Users may also want to create custom curriculum commands as well
# in this example, we will create a custom curriculum command called "noise"
# this will wrap a given environment in an image noise wrapper (noisy_env_helper.py)

# this functionality can be useful for adding types of curriculum blocks, such as
# interpolation between environments

def primary_interpreter(env_name, **kwargs):
	return gym.make(env_name)

# the key reader takes data and a duration, as well as other curriculum kwargs
# it returns a block of one or more environments to be added to the curriculum
def noisy_key_reader(data, duration, **kwargs):
	env, noise = data
	env_object = primary_interpreter(env)
	env_object = NoisyEnv(env_object, noise)

	block = [
		[env_object, duration]
	]

	return block



# now we can build our schedule using the special key "noise"
# the format is [ {key:data}, duration ]

schedule = [

	["BreakoutNoFrameskip-v4", 3],

	[{"noise":["PongNoFrameskip-v4", 0.2]}, 1],
	[{"noise":["PongNoFrameskip-v4", 0.4]}, 1],
	[{"noise":["PongNoFrameskip-v4", 0.6]}, 1]
]

# wrap this schedule into a single environment
# we pass in the key reader under 'key_readers'
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		primary_interpreter = primary_interpreter, ### tell curriculum_tools how to process names
		key_readers = {"noise":noisy_key_reader}, ### tell curriculum_tools how to process special keys
		episodic=True
	)



# play this out with a random agent
for ep in range(total_episodes):
	env.reset()
	done = False
	while not done:
		a = env.action_space.sample()
		state, reward, done, info = env.step(a)

		#render the state
		state = np.asarray(state)[:,:,::-1] #bgr -> rgb
		cv2.namedWindow('state', cv2.WINDOW_NORMAL)
		cv2.imshow('state', state)
		cv2.waitKey(1)
