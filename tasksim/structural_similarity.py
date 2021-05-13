import numpy as np
import ot
import ot.lp
from time import time as timer


DEFAULT_CA = 0.5
DEFAULT_CS = 0.995

# OPTIMAL = set c_s as close to 1 and c_a as close to 0 as still seems reasonable
# I like having c_a as 0.5, since it evenly balances d_rwd and d_emd, then c_s around 0.99-0.999ish
def compute_constant_limit(c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    # solving the following recurrence relations for a(n) and s(n):
    # a(n+1) = 1 - c_a(1 - s(n))
    # s(n+1) = c_s*a(n+1)
    # ...
    # a(n+1) = 1 - c_a(1 - c_s*a(n))
    # s(n+1) = c_s*(1 - c_a(1 - s(n)))
    A = c_s*c_a
    B = c_s - c_s*c_a
    #C = 1 - c_a
    limit_s = B/(1 - A) # Also just...(c_s - c_s*c_a)/(1 - c_s*c_a) = (c_s - A)/(1 - A) = (A - c_s)/(A - 1)
    #limit_a = C/(1 - A)
    return 1 - limit_s

def final_score(S):
    if isinstance(S, tuple):
        S = S[0]
    ns, nt = S.shape
    a = np.array([1/ns for _ in range(ns)])
    b = np.array([1/nt for _ in range(nt)])
    return ot.emd2(a, b, 1-S)

def normalize_score(score, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    limit = compute_constant_limit(c_a, c_s)
    # TODO: how to normalize? simple division? or some other asymptotic curve, maybe logarithmic? idk
    return 1 - (1 - score)/(1 - limit)

def directed_hausdorff_numpy(delta_a, N_u, N_v):
    """
    Directed Hausdorff distance using a distance matrix and the out-neighbors of state nodes u and v. Note the the full
        Hausdorff distance is given by:

        max(directed_hausdorff_numpy(delta_a, N_u, N_v), directed_hausdorff_numpy(delta_a, N_v, N_u))

    :param delta_a: (np.array) State distance matrix.
    :param N_u: (list or np.array) Out neighbors (action nodes) of state node u.
    :param N_v: (list or np.array) Out neighbors (action nodes) of state node v.
    :return: (float) Directed Hausdorff distance.
    """
    #return delta_a[np.array(N_u)].swapaxes(0, 1)[np.array(N_v)].swapaxes(0, 1).min(axis=1).max()
    # ensure they're already np arrays, then do min along axis 0, instead of swapping and min along axis 1
    # since it's very small (at most 5 action neighbors), use python min/max instead of numpy
    return max([min(x) for x in delta_a[N_u].T[N_v].T])


def structural_similarity(action_dists, reward_matrix, out_neighbors_S, c_a=DEFAULT_CA, c_s=DEFAULT_CS, stop_rtol=1e-3,
                          stop_atol=1e-4, max_iters=1e5):
    """
    Compute the structural similarity of an MDP graph as inspired by Wang et. al. paper:
        https://www.ijcai.org/Proceedings/2019/0511.pdf.
        Note: We still want to augment this to work for different numbers of states/actions for different MDPs
    :param action_dists: (np.array) P(s' | s, a) for each action node in the MDP graph.
    :param reward_matrix: (np.array) r(s, a) for each action node in the MDP graph.
    :param out_neighbors_S: (dict) Dictionary mapping the state nodes to their corresponding action nodes.
    :param c_a: (float) a constant in [0, 1] discounting the impact of the neighbors N_alpha and N_beta on the pair of state
        nodes (u, v). Typically set to 0.95.
    :param c_s: (float) parameter in [0, 1] to weight the importance of the reward similarity and the transition
        similarity. Typically set to 0.95.
    :param stop_rtol: (float) The rtol paramter to numpy.allclose. Used for convergence criteria.
    :param stop_atol: (float) The atol paramter to numpy.allclose. Used for convergence criteria.
    :param max_iters: (int) Maximum number of iterations to attempt to make distance matrices converge.
    :return: (np.array, np.array, int, bool) (S_star, A_star, num_iters, done) The state and action similarity matrices,
        respectively, then then number of iterations to convergence and a bool stating whether it actually converged or
        not.
    """

    n_actions, n_states = action_dists.shape

    S = np.eye(n_states)
    A = np.eye(n_actions)
    states = list(range(n_states))

    one_minus_c_a = 1 - c_a  # optimization
    emd_maxiters = 1e5
    #  1:1 2:1
    #  1:2 2:2
    #   eye(2n)

    last_S = S.copy()
    last_A = A.copy()

    # Can precompute expected rewards since loop invariant
    expected_rewards = np.einsum('ij,ij->i', action_dists, reward_matrix) 
    cached_reward_differences = np.zeros((len(expected_rewards), len(expected_rewards)))
    for i in range(len(expected_rewards)):
        for j in range(i + 1, len(expected_rewards)):
            diff = abs(expected_rewards[i] - expected_rewards[j])
            cached_reward_differences[i][j] = diff
            cached_reward_differences[j][i] = diff
    # Gotta figure out if we can precompute d_rwd and shit
    done = False
    iter = 0
    while not done and iter < max_iters:
        # TODO: some amount of parallelization
        one_minus_S = 1 - S
        for u in states:
            for v in states[u + 1:]:
                # Note: Could we just use the P matrix to know what the out_neighbors are for actions and states?
                for alpha in out_neighbors_S[u]:
                    for beta in out_neighbors_S[v]:
                        d_rwd = cached_reward_differences[alpha, beta]

                        x = action_dists[alpha]
                        y = action_dists[beta]

                        _, d_emd, _, _, _ = ot.lp.emd_c(x, y, one_minus_S, emd_maxiters)

                        #import pdb; pdb.set_trace()
                        entry = 1 - one_minus_c_a * d_rwd - c_a * d_emd
                        A[alpha, beta] = entry
                        A[beta, alpha] = entry  # have to keep track of whole matrix for directed hausdorff

        one_minus_A = 1 - A
        for u in states:
            for v in states[u + 1:]:
                if not len(out_neighbors_S[u]) or not len(out_neighbors_S[v]):
                    continue
                haus1 = directed_hausdorff_numpy(one_minus_A, out_neighbors_S[u], out_neighbors_S[v])
                haus2 = directed_hausdorff_numpy(one_minus_A, out_neighbors_S[v], out_neighbors_S[u])
                haus = max(haus1, haus2)
                entry = c_s * (1 - haus)
                S[u, v] = entry
                S[v, u] = entry  # Note: might be unnecessary, just need upper triangle of matrix due to symmetry

        if np.allclose(A, last_A, rtol=stop_rtol, atol=stop_atol) and np.allclose(S, last_S, rtol=stop_rtol,
                                                                                  atol=stop_atol):
            # Note: Could update this to use specified more specific convergence criteria
            done = True
        else:
            last_S = S.copy()
            last_A = A.copy()

        iter += 1

    return S, A, iter - 1, done  # return done to report that it converged (or didn't)


if __name__ == "__main__":
    # Example based on the MDP described in Figure 1 of https://www.ijcai.org/Proceedings/2019/0511.pdf.
    out_neighbors_S = {0: [0, 1], 1: [2, 3], 2: [], 3: [], 4: [], 5: []}
    P = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.2, 0.5, 0.3],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.15, 0.15, 0.25, 0.45]])
    R = np.array([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.6, 0.7, 0.9],
                  [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.2, 0.4, 0.6, 0.5]])

    c_s = 0.95
    c_a = 0.95
    s, a, num_iters, d = structural_similarity(P, R, out_neighbors_S, c_a, c_s)

    print("S\n", s)
    print("A\n", a)
    print("num iters:", num_iters)
    print("converged:", d)
