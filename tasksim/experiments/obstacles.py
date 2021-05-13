import numpy as np

import tasksim
from tasksim import MDPGraph
import tasksim.structural_similarity as sim
import tasksim.gridworld_generator as gen


def permute_single(grid, success_prob):
    rows, cols = grid.shape
    def augment(grid, i, j):
        ret = grid.copy()
        if ret[i, j] != 2:
            ret[i, j] = 1
        return ret
    grids =  np.array([[augment(grid, i, j) for j in range(cols)] for i in range(rows)])
    graphs = np.array([[MDPGraph.from_grid(grids[i, j], success_prob) for j in range(cols)] for i in range(rows)])
    return grids, graphs

def get_center_idxs(shape):
    rows, cols = shape
    return [(rows//2, cols//2)]

def get_corner_idxs(shape):
    rows, cols = shape
    return [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]

def get_edge_idxs(shape):
    rows, cols = shape
    ret = []
    for i in range(1, rows-1):
        ret.append((i, 0))
        ret.append((i, cols-1))
    for j in range(1, cols-1):
        ret.append((0, j))
        ret.append((rows-1, j))
    return ret



if __name__ == '__main__':
    random_state = np.random.RandomState(seed=314159)

    success_prob = 0.9
    shape3 = (3, 3)
    grids, graphs = permute_single(gen.create_grid(shape3), success_prob)
    base = get_center_idxs(shape3)[0]
    corners = get_corner_idxs(shape3)
    edges = get_edge_idxs(shape3)

    print('Base:')
    print(grids[base])
    print()

    print('Corners:')
    for c in corners:
        print(grids[c])
        print()

    print('Edges:')
    for e in edges:
        print(grids[e])
        print()

    def verify_extract(scores, stop_rtol=1e-3, stop_atol=1e-4):
        assert len(scores)
        assert np.allclose(scores, [scores[0]]*len(scores), rtol=stop_rtol, atol=stop_atol)
        return scores[0]
    base_to_base = [graphs[base].compare2_norm(graphs[base])]
    print('Base to Base:', verify_extract(base_to_base))
    base_to_corners = [graphs[base].compare2_norm(graphs[corner]) for corner in corners]
    print('Base to Corners:', verify_extract(base_to_corners))
    base_to_edges = [graphs[base].compare2_norm(graphs[edge]) for edge in edges]
    print('Base to Edges:', verify_extract(base_to_edges))
    corners_to_corners = [graphs[corner].compare2_norm(graphs[corner2]) for corner in corners for corner2 in corners]
    print('Corners to Corners:', verify_extract(corners_to_corners))
    corners_to_edges = [graphs[corner].compare2_norm(graphs[edge]) for corner in corners for edge in edges]
    print('Corners to Edges:', verify_extract(corners_to_edges))
    edges_to_edges = [graphs[edge].compare2_norm(graphs[edge2]) for edge in edges for edge2 in edges]
    print('Edges to Edges:', verify_extract(edges_to_edges))

