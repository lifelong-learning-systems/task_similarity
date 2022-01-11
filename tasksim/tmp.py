import tasksim
import tasksim.gridworld_generator as gen
import tasksim.structural_similarity as sim

import ot
import numpy as np

# grid1 = gen.create_grid((2, 2))
# grid2 = grid1.copy()
# grid2[0, 1] = 1
# G1 = gen.MDPGraph.from_grid(grid1, success_prob=1, reward=10, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS)
# G2 = gen.MDPGraph.from_grid(grid2, success_prob=1, reward=10, strat=gen.ActionStrategy.NOOP_EFFECT_COMPRESS)
# S, _ = G1.compare_song(G2)
# S1, _ = G1.compare_song(G1)
# S2, _ = G2.compare_song(G2)

# import pdb; pdb.set_trace()


tasksim.init_ray()

shape = (7, 7)
prob = 0.5
goal_locs = [(0, 6)]
STEP_REWARD = -0.01
STRAT = gen.ActionStrategy.NOOP_EFFECT_COMPRESS
def create(obs, goals, step_reward=STEP_REWARD, strat=STRAT):
    G = gen.MDPGraph.from_grid(gen.create_grid(shape, obstacle_locations=obs, goal_locations=goals), prob, strat=strat)
    G.R[G.R != 1] = step_reward
    return G

base_obs = []
base_g = create(base_obs, goal_locs)

simple_obs = [(3, 3)]
simple_g = create(simple_obs, goal_locs)

basic_obs = [(0, 5), (1, 5), (1, 6)]
basic_g = create(basic_obs, goal_locs)

full_obs = [(i, j) for i in range(7) for j in range(7) if not (i == 0 and j == 6)]
full_g = create(full_obs, goal_locs)

almost_obs = [(i, j) for i in range(7) for j in range(7) if not (i == 0 and j == 6 or i == 0 and j == 5)]
almost_g = create(almost_obs, goal_locs)

combined_obs = basic_obs + simple_obs
combined_g = create(combined_obs, goal_locs)

def display(g):
    print(g.grid)
    S, A, num_iters, done = base_g.compare(g)
    print(sim.final_score(S))
    print(base_g.compare2_song(g))
    print(num_iters)

print('Basic')
display(basic_g)

print()
print('Full')
display(full_g)

print()
print('Almost')
display(almost_g)

# UH OH: 
# Compare across: NOOP_EFFECT vs. NOOP_ACTION
# Compare across: 0 reward vs. -0.01 reward