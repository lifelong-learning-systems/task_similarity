import gym
import curriculum_tools
import cv2
import numpy as np

pong = gym.make("PongNoFrameskip-v4")
breakout = gym.make("BreakoutNoFrameskip-v4")

# play (pong and then breakout) 5 times
sub_schedule = [
	[pong, 1],
	[breakout, 1]
]

schedule = [
	[{"repeat":sub_schedule}, 5]
]

# wrap this schedule into a single environment
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
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
