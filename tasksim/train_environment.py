from ray import tune
import gym, ray
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import tasksim
from tasksim.environment import MDPGraphEnv
from ray.tune.schedulers import PopulationBasedTraining
import tasksim.gridworld_generator as gen


def env_creator(_):
    grid = gen.create_grid((10, 10), obstacle_prob=0.2)
    G = gen.MDPGraph.from_grid(grid, strat=gen.ActionStrategy.WRAP_NOOP_EFFECT)
    env = MDPGraphEnv(G)
    return env # return an env instance

register_env("gridworld", env_creator)

config = ppo.DEFAULT_CONFIG.copy()
print(config)
config.update({"env": "gridworld",
        'env_config':{'visualize':False},
        # "callbacks": {
        #     "on_train_result": on_train_result,
        # },

        'framework':'torch',
        "num_workers": 1,
        'model': {
                "use_lstm": False,
            }
        })



tune.run(PPOTrainer, config=config,
        checkpoint_freq = 1,
        name="gridworld",
)  # "log_level": "INFO" for verbose,
                                                     # "framework": "tfe"/"tf2" for eager,

