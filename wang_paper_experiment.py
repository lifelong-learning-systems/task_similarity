"""
Code to replicate the results from Wang paper (https://www.ijcai.org/Proceedings/2019/0511.pdf), section 5.1.
"""

import numpy as np
from structural_similarity import structural_similarity
from matplotlib import pyplot as plt
import seaborn as sns

# grid: where goal state == 4
# 6 7 8
# 3 4 5
# 0 1 2

a_map = {"left": 0, "right": 1, "up": 2, "down": 3}


def move_allowed(grid_shape, state, action):
    # if a move is allowed (has positive probability) for Wang paper example.
    if state == 4:  # goal state
        return False, None
    if action == "left":
        if state % grid_shape[1] == 0:
            return False, None
        else:
            s_p = state - 1
            return True, s_p
    elif action == "right":
        if state % grid_shape[1] == grid_shape[0] - 1:
            return False, None
        else:
            s_p = state + 1
            return True, s_p
    elif action == "up":
        if state // grid_shape[1] % grid_shape[0] == grid_shape[0] - 1:
            return False, None
        else:
            s_p = state + grid_shape[1]
            return True, s_p
    elif action == "down":
        if state < grid_shape[1]:
            return False, None
        else:
            s_p = state - grid_shape[1]
            return True, s_p


if __name__ == "__main__":
    P = np.zeros((9, 4))
    out_neighbors_s = dict(zip(range(9), [np.empty(shape=(0,), dtype=int) for i in range(9)]))
    out_neighbors_a = dict()

    a_node = 0
    for s in range(9):
        for act in list(a_map.keys()):
            move_allow, s_prime = move_allowed((3, 3), s, act)
            if move_allow:  # add action node
                out_neighbors_s[s] = np.append(out_neighbors_s[s], (a_node))
                out_neighbors_a[a_node] = np.zeros(9)
                for a_p in list(a_map.keys()):
                    if act == a_p:  # agent can go in the given direction
                        out_neighbors_a[a_node][s_prime] = 0.9
                        continue
                    allowed2, s_prime2 = move_allowed((3, 3), s, a_p)
                    if allowed2:  # agent can go in alternate direction as well
                        out_neighbors_a[a_node][s_prime2] = 0.1

                # If there are multiple alternate states besides the chosen action, normalize the probabilities between
                #   them to get a correct probability distribution
                if np.sum(out_neighbors_a[a_node]) > 1.0:
                    eq_01 = np.where(out_neighbors_a[a_node] == 0.1)
                    n_eq = len(eq_01[0])
                    out_neighbors_a[a_node][np.where(out_neighbors_a[a_node] == 0.1)] = 0.1 / n_eq
                a_node += 1

    P = np.array([arr for arr in out_neighbors_a.values()])
    R = np.zeros((20, 9))
    for i in range(20):
        if P[i, 4] > 0:
            R[i, 4] = 1.0

    print("State out neighbors dict:")
    print(out_neighbors_s)
    print("Action out neighbors dict:")
    print(out_neighbors_a)
    print("Action node transition probabilities:")
    print(P)
    print("Action node reward vectors:")
    print(R)

    # compute structural similarity
    sigma_s, sigma_a, num_iters, done = structural_similarity(P, R, out_neighbors_s)

    print("delta_S:")
    print(1 - sigma_s)

    print("Exact comparison value from paper:")
    print(f"delta_S(0, 2): Ours: {1 - sigma_s[0, 2]}, Paper: 0.27489")
    print(f"delta_S(0, 6): Ours: {1 - sigma_s[0, 6]}, Paper: 0.27489")
    print(f"delta_S(1, 3): Ours: {1 - sigma_s[1, 3]}, Paper: 0.27492")
    print(f"delta_S(1, 5): Ours: {1 - sigma_s[1, 5]}, Paper: 0.27492")
    print(f"delta_S(1, 7): Ours: {1 - sigma_s[1, 7]}, Paper: 0.28627")
    print(f"delta_S(0, 8): Ours: {1 - sigma_s[0, 8]}, Paper: 0.29873")

    print("Iterations to converge:", num_iters)

    upper = np.triu(sigma_s)

    ax = sns.heatmap(1 - upper, linewidth=0.5, cmap="rainbow")
    plt.show()






