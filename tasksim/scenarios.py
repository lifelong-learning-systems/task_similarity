from argparse import ArgumentError
from os import wait
import matplotlib
import tasksim
from tasksim.qtrainer import *
import time

def run_simulation(trainer : QTrainer, repeat=2, delay=0.1):
    trainer.learning = False
    for _ in range(repeat):
        obs = trainer.reset(center=False)
        steps = 0
        done = False
        while not done and steps < 100:
            
            plt.imshow(obs)
            plt.draw()
            plt.pause(delay)
            obs, rewards, done, _ = trainer.step(center=False)
            steps += 1
        

def sim_score(env1, env2, metric='new'):
    if metric == 'new':
        S, A, num_iters, _ = env1.graph.compare(env2.graph)
        return 1 - sim.final_score(S)            
    elif metric == 'song':
        S, A, num_iters, _ = env1.graph.compare(env2.graph)
        return 1 - sim.final_score(S)            
    else:
        raise ArgumentError(f'Invalid metric option: {metric}')


if __name__ == '__main__':
    
    
    
    env1 = EnvironmentBuilder((7, 7)) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.0) \
            .set_goals([ravel(0, 6)]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()
    
    env2 = EnvironmentBuilder((7, 7)) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles([ravel(x,y) for x in range(7) for y in range(7) if (x, y) != (6, 0)]) \
            .set_goals([ravel(0, 6)]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()

    
    
    trainer = QTrainer(env1, agent_path='test_agent.json')
    num_episodes, num_steps, performance = trainer.run(num_iters=5000, early_stopping=True, threshold=0.99)
    print("\nnum_episodes:", num_episodes, "\n num_steps:", num_steps, "\n")

    trainer2 = QTrainer(env2, agent_path='test_agent_transfer.json')
    
    obs = trainer2.reset()
    
    trainer.transfer_to(trainer2, direct_transfer=True)
    
    num_episodes_transfer, num_steps_transfer, performance_transfer = trainer2.run(num_iters=1500, early_stopping=True, threshold=0.99)
    print("\nnum_episodes_transfer:", num_episodes_transfer, "\n num_steps_transfer:", num_steps_transfer, "\n")

    #run_simulation(trainer, delay=0.35)

    #run_simulation(trainer2, delay=0.35)

    print("similarity_score: ", sim_score(env1, env2))