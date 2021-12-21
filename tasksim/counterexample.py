from os import wait
from gym.core import Env
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
        

def sim_score(env1, env2):
    S, A, num_iters, _ = env1.graph.compare(env2.graph)
    return 1 - sim.final_score(S)


if __name__ == "__main__":

    open_env = EnvironmentBuilder((7, 7)) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.0) \
            .set_goals([ravel(0, 6)]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()

    envs = []
    envs.append(open_env)
    sim_scores = []
    for goal in range(open_env.grid.shape[0] * open_env.grid.shape[1]):
        if unravel(goal) == (6, 0) or unravel(goal) == (6, 1) or unravel(goal) == (5, 1) or unravel(goal) == (5, 0):
            continue
        
        obstacles = []
        lobs = tuple(np.add((0, -1), unravel(goal)))
        if lobs[1] >= 0:
            obstacles.append(ravel(*lobs))

        robs = tuple(np.add((0, 1), unravel(goal)))
        if(robs[1] <= 6):
            obstacles.append(ravel(*robs))

        uobs = tuple(np.add((-1, 0), unravel(goal)))
        if(uobs[0] >= 0):
            obstacles.append(ravel(*uobs))

        dobs = tuple(np.add((1, 0), unravel(goal)))
        if(dobs[0] <= 6):
            obstacles.append(ravel(*dobs))
        
        

        env = EnvironmentBuilder((open_env.grid.shape[0], open_env.grid.shape[1])) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacles) \
            .set_goals([goal]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()

        envs.append(env)



    for env in envs:
        sim_scores.append(sim_score(open_env, env))
    
    # def isqrt(n):
    #     x = n
    #     y = (x + 1) // 2
    #     while y < x:
    #         x = y
    #         y = (x + n // x) // 2
    #     return x

    

    fig, axs = plt.subplots(7, 7, figsize=(25, 25))
    
    first = True
    for ax, env, score in zip(axs.flatten(), envs, sim_scores):
        ax.imshow(env.reset(center=False))
        if first:
            ax.set_title("ORIGINAL GRID")
            first = False
            continue
        simscore = "Similarity = " + str(score)
        ax.set_title(simscore)


    other_env = EnvironmentBuilder((open_env.grid.shape[0], open_env.grid.shape[1])) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.05) \
            .set_goals([3]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()

    other_env2 = EnvironmentBuilder((open_env.grid.shape[0], open_env.grid.shape[1])) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.05) \
            .set_goals([3, 5, 9]) \
            .set_success_prob(1.0) \
            .set_fixed_start(ravel(6, 0)) \
            .build()

    other_env3 = EnvironmentBuilder((open_env.grid.shape[0], open_env.grid.shape[1])) \
            .set_strat(gen.ActionStrategy.NOOP_EFFECT)\
            .set_obstacles(obstacle_prob=0.05) \
            .set_goals([0]) \
            .set_success_prob(1.0) \
            .build()

    axs.flatten()[-3].imshow(other_env.reset(center=False))
    simscore = "Similarity = " + str(sim_score(open_env, other_env))
    axs.flatten()[-3].set_title(simscore)

    axs.flatten()[-2].imshow(other_env2.reset(center=False))
    simscore = "Similarity = " + str(sim_score(open_env, other_env2))
    axs.flatten()[-2].set_title(simscore)

    axs.flatten()[-1].imshow(other_env3.reset(center=False))
    simscore = "Similarity = " + str(sim_score(open_env, other_env3))
    axs.flatten()[-1].set_title(simscore)

    fig.tight_layout(pad=3.0)
    plt.show()


    