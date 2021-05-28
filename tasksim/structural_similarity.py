import numpy as np
import ot
import ot.lp
from time import time as timer
from sklearn.preprocessing import normalize, minmax_scale
from enum import Enum
import ray

from tasksim import NUM_CPU, emd_c, emd_c_chunk
import process_chunk as pc
from numba import jit
from numba.extending import get_cython_function_address
import ctypes
from ctypes import CFUNCTYPE, POINTER, c_double, c_int
import numpy.ctypeslib as npct

DEFAULT_CA = 0.5
DEFAULT_CS = 0.995

class InitStrategy(Enum):
    ZEROS = 1
    ONES = 2
    IDENTITY = 3

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

def truncate_score(score, num_decimal=3):
    mul = 10**num_decimal
    return np.trunc(score * mul)/(mul)

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



#@jit(nopython=True)
#@jit()
def compute_a_py(chunk, reward_diffs, actions1, actions2, one_minus_S, c_a, emd_maxiters):
    n_a1 = chunk.shape[0]
    n_a2 = chunk.shape[1]
    count = -1
    entries = np.ones(n_a1*n_a2) * -1
    
    W = one_minus_S.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    for i in range(n_a1):
        for j in range(n_a2):
            count += 1
            if (chunk[i, j][0] < 0) or (chunk[i, j][1] < 0):
                entries[count] = -1
                continue
            alpha = int(chunk[i, j][0])
            beta = int(chunk[i, j][1])
            d_rwd = reward_diffs[alpha, beta]
            x = actions1[alpha, :]
            y = actions2[beta, :]
            d_emd = emd_c(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          W, len(x), len(y), int(emd_maxiters))
            entry = 1 - (1 - c_a) * d_rwd - c_a * d_emd
            entries[count] = entry
    return entries

def compute_a_py_full(chunk, reward_diffs, actions1, actions2, one_minus_S, c_a, emd_maxiters):
    n_chunk1 = chunk.shape[0]
    n_chunk2 = chunk.shape[1]
    num_actions1, num_states1 = actions1.shape
    num_actions2, num_states2 = actions2.shape
    entries = np.zeros(n_chunk1*n_chunk2)
    def to_ptr(x):
        return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    emd_c_chunk(n_chunk1, n_chunk2, num_actions1, num_actions2, num_states1, num_states2,
                          to_ptr(chunk), to_ptr(reward_diffs), to_ptr(actions1), to_ptr(actions2), to_ptr(one_minus_S),
                          to_ptr(entries),
                          np.float64(c_a), int(emd_maxiters))
    return entries


@ray.remote
def compute_a(chunk, reward_diffs, actions1, actions2, one_minus_S, c_a, emd_maxiters, handle=pc.compute_chunk):
    return handle(chunk, reward_diffs, actions1, actions2, one_minus_S,
                            np.float64(c_a), np.float64(emd_maxiters))

# TODO: combine common functionality between SS and CSS
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
    return cross_structural_similarity(action_dists, action_dists, reward_matrix, reward_matrix,
                                       out_neighbors_S, out_neighbors_S,
                                       c_a=c_a, c_s=c_s, stop_rtol=stop_rtol, stop_atol=stop_atol, max_iters=max_iters,
                                       self_similarity=True)

# TODO: also consider (lower importance) other granularities besides each state (subgraph???)
def cross_structural_similarity(action_dists1, action_dists2, reward_matrix1, reward_matrix2, out_neighbors_S1,
                                out_neighbors_S2, c_a=DEFAULT_CA, c_s=DEFAULT_CS, stop_rtol=1e-3,
                                stop_atol=1e-4, max_iters=1e5,
                                init_strategy: InitStrategy = InitStrategy.ZEROS, self_similarity=False):
    n_actions1, n_states1 = action_dists1.shape
    n_actions2, n_states2 = action_dists2.shape

    # Initialization SHOULDN'T matter that much...
    # zeros means normalizing slightly overshoots (positive)
    # ones means normalizing slightly undershoots (negative, weird)
    if init_strategy == InitStrategy.IDENTITY or self_similarity:
        S = np.zeros((n_states1, n_states2))
        A = np.zeros((n_actions1, n_actions2))
        np.fill_diagonal(S, 1)
        np.fill_diagonal(A, 1)
    elif init_strategy == InitStrategy.ZEROS:
        S = np.zeros((n_states1, n_states2))
        A = np.zeros((n_actions1, n_actions2))
    else:
        S = np.ones((n_states1, n_states2))
        A = np.ones((n_actions1, n_actions2))

    states1 = list(range(n_states1))
    states2 = list(range(n_states2))

    one_minus_c_a = 1 - c_a  # optimization
    emd_maxiters = 1e5
    #  1:1 2:1
    #  1:2 2:2
    #   eye(2n)

    last_S = S.copy()
    last_A = A.copy()

    def norm(mat1, mat2, method):
        combined = np.concatenate([mat1.flatten(), mat2.flatten()])
        if method == 'l1' or method == 'l2':
            combined = normalize([combined], norm=method).squeeze()
        if method == 'minmax':
            combined = minmax_scale(combined)
        ret1 = combined[:mat1.size].reshape(mat1.shape)
        ret2 = combined[mat1.size:].reshape(mat2.shape)
        return ret1, ret2

    reward_matrix1, reward_matrix2 = norm(reward_matrix1, reward_matrix2, 'minmax')

    # Can precompute expected rewards since loop invariant
    expected_rewards1 = np.einsum('ij,ij->i', action_dists1, reward_matrix1) 
    expected_rewards2 = np.einsum('ij,ij->i', action_dists2, reward_matrix2) 
    def compute_diff(list1, list2):
        diff = np.zeros((len(list1), len(list2)))
        for i in range(len(list1)):
            for j in range(len(list2)):
                diff[i][j] = abs(list1[i] - list2[j])
        return diff
    cached_reward_differences = compute_diff(expected_rewards1, expected_rewards2)
    reward_diffs_id = ray.put(cached_reward_differences)
    actions1_id = ray.put(action_dists1)
    actions2_id = ray.put(action_dists2)


    if not self_similarity:
        action_pairs = np.array([np.float64((i, j)) for i in range(n_actions1) for j in range(n_actions2)]).reshape((n_actions1, n_actions2, 2))
    else:
        action_pairs = -1*np.ones((n_actions1, n_actions2, 2))
        for u in states1:
            for v in states2[u + 1:]:
                for alpha in out_neighbors_S1[u]:
                    for beta in out_neighbors_S2[v]:
                        action_pairs[alpha, beta, :] = np.float64((alpha, beta))

    action_chunks = np.array_split(action_pairs, NUM_CPU)

    done = False
    iter = 0
    while not done and iter < max_iters:
        # TODO: some amount of parallelization
        one_minus_S = 1 - S
        one_minus_S_id = ray.put(one_minus_S)

        # bind_compute_a = lambda chunk: compute_a.remote(chunk, reward_diffs_id, actions1_id, actions2_id, one_minus_S_id,
        #                                                 c_a, emd_maxiters,
        #                                                 handle=compute_a_py_full)
        #                                                 # handle=pc.compute_chunk)
        # remoted = [bind_compute_a(chunk) for chunk in action_chunks]
        # new_A = np.concatenate([ray.get(x) for x in remoted]).reshape((n_actions1, n_actions2))
        bind_compute_a = lambda chunk: compute_a_py_full(chunk, cached_reward_differences,
                                                    action_dists1, action_dists2,
                                                    one_minus_S, c_a, emd_maxiters)
        remoted = [bind_compute_a(chunk) for chunk in action_chunks]
        new_A = np.concatenate(remoted).reshape((n_actions1, n_actions2))
        A[new_A >= 0] = new_A[new_A >= 0]
        # Make symmetric if needed
        if self_similarity:
            i_lower = np.tril_indices(len(A), -1)
            A[i_lower] = A.T[i_lower]

        one_minus_A = 1 - A
        one_minus_A_transpose = one_minus_A.T
        for u in states1:
            if not self_similarity:
                v_list = states2
            else:
                v_list = states2[u+1:]
            for v in v_list:
                if not len(out_neighbors_S1[u]) or not len(out_neighbors_S2[v]):
                    continue
                haus1 = directed_hausdorff_numpy(one_minus_A, out_neighbors_S1[u], out_neighbors_S2[v])
                haus2 = directed_hausdorff_numpy(one_minus_A_transpose, out_neighbors_S2[v], out_neighbors_S1[u])
                haus = max(haus1, haus2)
                entry = c_s * (1 - haus)
                S[u, v] = entry
                if self_similarity:
                    S[v, u] = entry

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
