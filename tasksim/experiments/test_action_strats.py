import tasksim
import tasksim.gridworld_generator as gen

if __name__ == '__main__':
    grid = gen.create_grid((3, 3))
    grid[0][0] = 1

    G_subset = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.SUBSET)
    G_action = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.NOOP_ACTION)
    G_effect = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.NOOP_EFFECT)
    G_wrap_subset = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.WRAP_SUBSET)
    G_wrap_action = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.WRAP_NOOP_ACTION)
    G_wrap_effect = gen.MDPGraph.from_grid(grid, 0.9, strat=gen.ActionStrategy.WRAP_NOOP_EFFECT)

    graphs = [G_subset, G_action, G_effect, G_wrap_subset, G_wrap_action, G_wrap_effect]
    assert all([(G.P.sum(axis=1) == 1).all() for G in graphs]), 'All graphs prob dists should sum to 1'

    # TODO: resolve what happens when multiple actions go to the SAME state
    base_actions = G_subset.available_actions
    assert (base_actions[:, -1] == 0).all(), 'SUBSET should have no noop actions'
    noop_actions = G_action.available_actions
    assert (noop_actions[:, -1] == 1).all(), 'NOOP_ACTION should have all states with a noop action'
    assert (base_actions[:, :-1] == noop_actions[:, :-1]).all(), 'SUBSET & NOOP_ACTION should have all other actions the same'
    effect_actions = G_effect.available_actions
    assert (effect_actions[:, :-1] == 1).all() and (effect_actions[:, -1] == 0).all(), 'NOOP_EFFECT should have all actions available, no noop actions'