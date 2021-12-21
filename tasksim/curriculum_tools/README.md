
Curriculum Tools
========

Curriculum Tools is a package providing utilities for organizing gym environments into curricula, which can then be interacted with as a standard, single gym environment.


### Installation

Curriculum Tools is a pip package. Simply clone the repository and install with:
```
pip install -e .
```

The examples additionally depend on the OpenAI Gym Atari environments, which can be installed via:
```
pip install "gym[atari]"
```

### Basic Curricula

The basic form of a curriculum is a list of environments, along with a duration for each to be used for. This can be specified in either steps or full episodes. The make_curriculum utility will wrap this schedule into a single gym environment, which iterates through scheduled tasks as it is interacted with.  For example:

```python
import gym
import curriculum_tools
# create gym environments
pong = gym.make("PongNoFrameskip-v4")
breakout = gym.make("BreakoutNoFrameskip-v4")

# build a curriculum
schedule = [
	[pong, 3],
	[breakout, 3]
]

# compile into a single gym environment, and note the total curriculum duration
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True
	)
```

The above code creates a single environment object which tracks two gym environments behind the scenes. Once pong has been played for 3 episodes, it will automatically switch over to breakout, for an additional 3 episodes.

See `1_basic_example.py` for a complete example.

A similar curriculum can be constructed with steps as the unit of time:

```python
import gym
import curriculum_tools
# create gym environments
pong = gym.make("PongNoFrameskip-v4")
breakout = gym.make("BreakoutNoFrameskip-v4")

# build a curriculum
schedule = [
	[pong, 5000],
	[breakout, 5000]
]

# compile into a single gym environment, and note the total curriculum duration
env, total_steps = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=False
	)
```

In this case, pong is played for 5000 steps before it is switched out for breakout. Note that the behavior is to force pong to terminate at a specific step by overriding its "done" value.  This is inconsequential when playing many episodes, but is important to remember in shorter curricula or testing curricula.  If this behavior is undesirable, consider using episodes as the basis for your curricula.

### Advanced Curricula

In addition to sequential tasks, there are some specialized curricula entries that have additional capability. These are added with the following format:

```python
[{"key":<parameters>}, duration]
```

The environment is replaced with a dictionary having a single name (the key), and a corresponding value (parameters for this entry).


#### Task Pooling

Format: ````[{"pool":[config1, config2..., configN]}, duration]````

The ***pool*** key instructs the curriculum to maintain several environments and randomly choose among them at each reset() call. This is useful for learning several tasks at once or randomly sampling a subset of tasks.  For example:

```python
import gym
import curriculum_tools

# define pong and breakout
pong = gym.make("PongNoFrameskip-v4")
breakout = gym.make("BreakoutNoFrameskip-v4")

# play pong and breakout sampled randomly for 10 episodes
schedule = [
	[{"pool":[pong, breakout]}, 10]
]

# wrap this schedule into a single environment
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True
	)
```

See `2_pool_example.py` for a complete example.

#### Sub-Curricula Repetition

Format: ````[{"repeat":sub_curriculum}, number_of_times_to_repeat]````

To aid with building large curricula that may interweave past experience, the ***repeat*** key specifies an entire sub-curriculum to be repeated a certain number of times. The format of the sub-curriculum is identical to a typical curriculum, and may include special keys as well.  The duration of this entry indicates the number of times to repeat the sequence. For example:

```python
import gym
import curriculum_tools
# define pong and breakout
pong = gym.make("PongNoFrameskip-v4")
breakout = gym.make("BreakoutNoFrameskip-v4")

# define a sub-curriculum to repeat
sub_schedule = [
	[pong, 1],
	[breakout, 1]
]

# repeat 5 times, for 10 episodes total
schedule = [
	[{"repeat":sub_schedule}, 5]
]

# wrap this schedule into a single environment
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True
	)
```

See `3_repeat_example.py` for a complete example.

#### Custom Interpreters

A custom interpreter can be used to allow environments to be built dynamically, rather than ahead of time. This allows, for example, a schedule to be made with just the names of the desired environments:

```python
import gym
import curriculum_tools

def primary_interpreter(env_name, **kwargs):
    return gym.make(env_name)

schedule = [
	["PongNoFrameskip-v4", 3],
	["BreakoutNoFrameskip-v4", 3]
]

env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		primary_interpreter = primary_interpreter,
		episodic=True
	)

```

See `4_custom_interpreter.py` for a complete example.

### Custom Key Readers
Format: ````[{"custom-key": data_required}, duration]````

Users may also want to create custom curriculum commands as well.
This functionality can be useful for adding types of curriculum blocks, such as interpolation between environments.

In the full example `5_key_readers.py`, we create a custom curriculum command called "noise", that wraps a given environment in an image noise wrapper (`noisy_env_helper.py`)

```python
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

# This custom key reader then gets used in the schedule
schedule = [

	["BreakoutNoFrameskip-v4", 3],

	[{"noise":["PongNoFrameskip-v4", 0.2]}, 1],
	[{"noise":["PongNoFrameskip-v4", 0.4]}, 1],
	[{"noise":["PongNoFrameskip-v4", 0.6]}, 1]
]

env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		primary_interpreter = primary_interpreter,
		key_readers = {"noise":noisy_key_reader},
		episodic=True
	)
```


### Task ID Extraction

When maintaining a curriculum of many environments it is often necessary to query for an identifier of the current task. The curriculum_tools `NamedEnv` wrapper adds the ability to name an environment. This can be queried at any time with:

```python
env.unwrapped.name
```

You are free to use something other than a string to name the environment, such as a one-hot vector or some other identifying data.

For example:

```python
import gym
import curriculum_tools
from curriculum_tools import NamedEnv
pong = gym.make("PongNoFrameskip-v4")
pong = NamedEnv(pong, [1,0])
breakout = gym.make("BreakoutNoFrameskip-v4")
breakout = NamedEnv(breakout, [0,1])

schedule = [
	[pong, 3],	
	[breakout, 3],			   
]
env, total_episodes = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True
	)

for e in range(total_episodes):
	# curriculum advances on reset()
    env.reset()

	# what task are we running now?
    print("Running task", env.unwrapped.name)

    # run the episode
    done = False
    while not done:
        a = env.action_space.sample()
        s, r, done, info = env.step(a)
```

Outputs:

```python
Running task [1, 0]
Running task [1, 0]
Running task [1, 0]
Running task [0, 1]
Running task [0, 1]
Running task [0, 1]
```



### Parallel Training Support

The logistics of using distributed training with curricula can be difficult to manage. Our recommended solution is to re-build the curriculum in each worker thread, and divide the durations of all curriculum entries by the number of workers. This can be done automatically with the ````across_workers```` option:

```python
schedule = [
	[pong, 500],
	[breakout, 500]			   
]

# curriculum for single thread
env, total_duration = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True
	)

print(total_duration) # prints 1000

# curriculum for 4 workers in parallel
env, total_duration = curriculum_tools.make_curriculum(
		schedule=schedule,
		episodic=True,
        across_workers=4
	)

print(total_duration) # prints 250
```

In the example above, the duration is specified as 1000 steps across 4 workers. The curriculum that is built will use 250 steps (125 for each task), and assumes that four total copies of the curriculum exist. If your workers are synchronized on a step-wise basis, this means that 1000 steps will be experienced overall (500 of each task, as specified in the curriculum).

If the number of specified episodes does not evenly divide by the specified number of workers, the truncated result of the division will be used (i.e. if 8 workers are specified, then `total_duration` will be 124).
