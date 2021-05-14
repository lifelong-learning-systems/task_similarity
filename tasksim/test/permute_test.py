import numpy as np

import tasksim
from tasksim.gridworld_generator import create_grid, MDPGraph, compare_grid_symmetry, permute_grid


if __name__ == '__main__':
    grid = create_grid((3, 3))
    permuted = permute_grid(grid, keep_isomorphisms=True)
    center = [permuted[4]]
    corners = [permuted[0], permuted[2], permuted[6], permuted[8]]
    edges = [permuted[1], permuted[3], permuted[5], permuted[7]]

    def compare_lists(list1, list2):
        return [compare_grid_symmetry(a, b) for a in list1 for b in list2]
    val = compare_lists(center, center)
    assert np.all(val)
    print('center-center:', val[0])
    val = compare_lists(center, corners)
    assert not np.any(val)
    print('center-corners:', val[0])
    val = compare_lists(center, edges)
    assert not np.any(val)
    print('center-edges:', val[0])
    val = compare_lists(edges, edges)
    assert np.all(val)
    print('edges-edges:', val[0])
    val = compare_lists(edges, corners)
    assert not np.any(val)
    print('edges-corners:', val[0])
    val = compare_lists(corners, corners)
    assert np.all(val)
    print('corners-corners:', val[0])

    all_diff = [center[0], corners[0], edges[0]]
    filtered = permute_grid(grid, keep_isomorphisms=False)
    assert len(all_diff) == len(filtered)

