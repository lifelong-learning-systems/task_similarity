import numpy as np
import ot
import ot.lp
import math
from time import time as timer
from sklearn.preprocessing import normalize, minmax_scale
from enum import Enum
import ray
import scipy

from tasksim.utils import emd_c, emd_c_chunk, get_num_cpu, init_ray
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
    RAND = 4

# TODO: normalize by dividing by a constant?
# OPTIMAL = set c_s as close to 1 and c_a as close to 0 as still seems reasonable
# I like having c_a as 0.5, since it evenly balances d_rwd and d_emd, then c_s around 0.99-0.999ish
# TODO: incorporate rtol & atol by being inspired by numpy.isclose
# - isclose(a, b, rtol, btol) -> abs(a - b) <= (atol + rtol*abs(b))
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

def final_score(S, c_n=1.0):
    if isinstance(S, tuple):
        S = S[0]
    ns, nt = S.shape
    a = np.array([1/ns for _ in range(ns)])
    b = np.array([1/nt for _ in range(nt)])
    #d_num_states = (1 - 1/(abs(ns - nt) + 1))
    d_num_states = 1 - min(ns/nt, nt/ns)
    return c_n * ot.emd2(a, b, 1-S) + (1 - c_n) * d_num_states
def final_score_song(S):
    ns, nt = S.shape
    a = np.array([1/ns for _ in range(ns)])
    b = np.array([1/nt for _ in range(nt)])
    return 1 - ot.emd2(a, b, 1 - S)

def normalize_score(score, c_a=DEFAULT_CA, c_s=DEFAULT_CS):
    # Represents smallest possible distance metric given c_a, c_s (e.g. 0.15), with actual score of 0.16
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

@ray.remote
def compute_s(chunk, out_S1, out_S2, one_minus_A, one_minus_A_transpose, c_s):
    chunk_n1, chunk_n2, _ = chunk.shape
    count = -1
    entries = np.ones(chunk_n1 * chunk_n2) * -1
    for i in range(chunk_n1):
        for j in range(chunk_n2):
            count += 1
            if (chunk[i, j][0] < 0) or (chunk[i, j][1] < 0):
                entries[count] = -1
                continue
            u = int(chunk[i, j][0])
            v = int(chunk[i, j][1])
            if not len(out_S1[u]) or not len(out_S2[v]):
                # obstacle-obstacle, goal-goal, obstacle-goal, goal-obstacle all have MAX similarity
                # obstacle-normal, goal-normal all have MIN similarity
                if not len(out_S1[u]) and not len(out_S2[v]):
                    #entry = c_s
                    entry = c_s
                else:
                    entry = 0
            else:
                haus1 = directed_hausdorff_numpy(one_minus_A, out_S1[u], out_S2[v])
                haus2 = directed_hausdorff_numpy(one_minus_A_transpose, out_S2[v], out_S1[u])
                haus = max(haus1, haus2)
                # action_idx1 = out_S1[u]
                # action_idx2 = out_S1[v]
                # grid_idx = np.meshgrid(action_idx1, action_idx2)
                # haus = ot.lp.emd2(action_idx1, action_idx2, one_minus_A[grid_idx])
                entry = c_s * (1 - haus)
            entries[count] = entry
    return entries


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

def norm(mat1, mat2, method):
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    combined = np.concatenate([mat1.flatten(), mat2.flatten()])
    if method == 'l1' or method == 'l2':
        combined = normalize([combined], norm=method)[0]
    if method == 'minmax':
        combined = minmax_scale(combined)
    ret1 = combined[:mat1.size].reshape(mat1.shape)
    ret2 = combined[mat1.size:].reshape(mat2.shape)
    return ret1, ret2



def cross_structural_similarity_song(action_dists1, action_dists2, reward_matrix1, reward_matrix2, out_neighbors_S1, out_neighbors_S2,
                                     c=DEFAULT_CA, stop_tol=1e-4, max_iters=1e5):
    _, n_states1 = action_dists1.shape
    _, n_states2 = action_dists2.shape
    states1 = list(range(n_states1))
    states2 = list(range(n_states2))

    d = np.zeros((n_states1, n_states2))
    d_prime = np.zeros((n_states1, n_states2))
    delta = np.inf
    # Paper assumes s' doesn't matter for reward; since our MDPs are stochastic, it does matter;
    # For fair comparison, using same reward function as other metric
    def compute_exp(P, R):
        n = P.shape[0]
        # TODO for paper: describe it as discretizing reward distribution, capturing more than just expected value
        ret_pos = np.zeros((n,))
        ret_neg = np.zeros((n,))
        for i in range(n):
            probs, rewards = P[i], R[i]
            pos, neg = rewards >= 0, rewards < 0
            ret_pos[i] = np.dot(probs[pos], rewards[pos])
            ret_neg[i] = np.dot(probs[neg], rewards[neg])
        return ret_pos, ret_neg
    expected_rewards_pos1, expected_rewards_neg1 = compute_exp(action_dists1, reward_matrix1)
    expected_rewards_pos2, expected_rewards_neg2 = compute_exp(action_dists2, reward_matrix2)
    def compute_diff(pos1, pos2, neg1, neg2):
        diff = np.zeros((len(pos1), len(pos2)))
        for i in range(len(pos1)):
            for j in range(len(pos2)):
                diff[i][j] = 0.5*abs(pos1[i] - pos2[j]) + 0.5*abs(neg1[i] - neg2[j])
        return diff
    cached_reward_differences = compute_diff(expected_rewards_pos1, expected_rewards_pos2, expected_rewards_neg1, expected_rewards_neg2)

    # TODO: add performance optimizations as needed
    while not (delta < stop_tol):
        for s_i, s_j in zip(states1, states2):
            actions_i = out_neighbors_S1[s_i]
            actions_j = out_neighbors_S2[s_j]
            # Limited to paper specifics
            assert len(actions_i) == len(actions_j), 'For Song metric, actions must have 1:1 correspondence'
            # Assumes 1:1 correspondence in effect
            for a_i, a_j in zip(actions_i, actions_j):
                P_a_i = action_dists1[a_i]
                P_a_j = action_dists2[a_j]
                # TODO: If determinstic, this gets simplified?
                # d_emd = d[s_i, s_j]
                d_emd = emd_c(P_a_i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              P_a_j.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                              d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(P_a_i), len(P_a_j), int(max_iters))
                d_rwd = cached_reward_differences[a_i, a_j]
                # TODO: what if this is greater than 1? Is that still okay?? Should we scale d_rwd by (1-c)?
                #import pdb; pdb.set_trace()
                tmp = (1 - c)*d_rwd + c*d_emd
                assert tmp <= 1, 'd_rwd & d_emd combination resulted in value greater than 1'
                d_prime[s_i, s_j] = max(d_prime[s_i, s_j], tmp)
        delta = np.sum(np.abs(d_prime - d)) / (n_states1*n_states2)
        for s_i, s_j in zip(states1, states2):
            d[s_i, s_j] = d_prime[s_i, s_j]
    return d


# TODO: Change InitStrategy to be ONES? Would need to re-run experiments...
def cross_structural_similarity(action_dists1, action_dists2, reward_matrix1, reward_matrix2, out_neighbors_S1,
                                out_neighbors_S2, c_a=DEFAULT_CA, c_s=DEFAULT_CS, stop_rtol=1e-3,
                                stop_atol=1e-4, max_iters=1e5,
                                init_strategy: InitStrategy = InitStrategy.RAND, self_similarity=False):
    cpus = get_num_cpu()
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
    elif init_strategy == InitStrategy.ONES:
        S = np.ones((n_states1, n_states2))
        A = np.ones((n_actions1, n_actions2))
    else:
        rng = np.random.default_rng(seed=123)
        S = rng.random((n_states1, n_states2))
        A = rng.random((n_actions1, n_actions2))


    states1 = list(range(n_states1))
    states2 = list(range(n_states2))

    one_minus_c_a = 1 - c_a  # optimization
    emd_maxiters = 1e5
    #  1:1 2:1
    #  1:2 2:2
    #   eye(2n)

    last_S = S.copy()
    last_A = A.copy()


    # TODO: handle negative rewards differently...should be able to handle [-1, 1] range
    #reward_matrix1, reward_matrix2 = norm(reward_matrix1, reward_matrix2, 'minmax')

    def compute_exp(P, R):
        n = P.shape[0]
        # TODO for paper: describe it as discretizing reward distribution, capturing more than just expected value
        ret_pos = np.zeros((n,))
        ret_neg = np.zeros((n,))
        for i in range(n):
            probs, rewards = P[i], R[i]
            pos, neg = rewards >= 0, rewards < 0
            ret_pos[i] = np.dot(probs[pos], rewards[pos])
            ret_neg[i] = np.dot(probs[neg], rewards[neg])
        return ret_pos, ret_neg
    expected_rewards_pos1, expected_rewards_neg1 = compute_exp(action_dists1, reward_matrix1)
    expected_rewards_pos2, expected_rewards_neg2 = compute_exp(action_dists2, reward_matrix2)
    def compute_diff(pos1, pos2, neg1, neg2):
        diff = np.zeros((len(pos1), len(pos2)))
        for i in range(len(pos1)):
            for j in range(len(pos2)):
                diff[i][j] = 0.5*abs(pos1[i] - pos2[j]) + 0.5*abs(neg1[i] - neg2[j])
        return diff
    cached_reward_differences = compute_diff(expected_rewards_pos1, expected_rewards_pos2, expected_rewards_neg1, expected_rewards_neg2)
    reward_diffs_id = ray.put(cached_reward_differences)
    actions1_id = ray.put(action_dists1)
    actions2_id = ray.put(action_dists2)
    out_S1_id = ray.put(out_neighbors_S1)
    out_S2_id = ray.put(out_neighbors_S2)

    if not self_similarity:
        action_pairs = np.array([np.float64((i, j)) for i in range(n_actions1) for j in range(n_actions2)]).reshape((n_actions1, n_actions2, 2))
        state_pairs = np.array([np.float64((i, j)) for i in range(n_states1) for j in range(n_states2)]).reshape((n_states1, n_states2, 2))
    else:
        action_pairs = -1*np.ones((n_actions1, n_actions2, 2))
        state_pairs = -1*np.ones((n_states1, n_states2, 2))
        for u in states1:
            for v in states2[u + 1:]:
                state_pairs[u, v, :] = np.float64((u, v))
                for alpha in out_neighbors_S1[u]:
                    for beta in out_neighbors_S2[v]:
                        action_pairs[alpha, beta, :] = np.float64((alpha, beta))

    action_chunks = np.array_split(action_pairs, cpus)
    state_chunks = np.array_split(state_pairs, cpus)

    done = False
    iter = 0
    while not done and iter < max_iters:
        # TODO: some amount of parallelization
        one_minus_S = 1 - S
        one_minus_S_id = ray.put(one_minus_S)

        bind_compute_a = lambda chunk: compute_a.remote(chunk, reward_diffs_id, actions1_id, actions2_id, one_minus_S_id,
                                                        c_a, emd_maxiters,
                                                        handle=compute_a_py_full)
                                                        # handle=pc.compute_chunk)
        remoted_a = [bind_compute_a(chunk) for chunk in action_chunks]
        new_A = np.concatenate([ray.get(x) for x in remoted_a]).reshape((n_actions1, n_actions2))
        A[new_A >= 0] = new_A[new_A >= 0]
        # Make symmetric if needed
        if self_similarity:
            i_lower = np.tril_indices(len(A), -1)
            A[i_lower] = A.T[i_lower]

        one_minus_A = 1 - A
        one_minus_A_transpose = one_minus_A.T
        one_minus_A_id = ray.put(one_minus_A)
        one_minus_A_transpose_id = ray.put(one_minus_A.T)
        bind_compute_s = lambda chunk: compute_s.remote(chunk, out_S1_id, out_S2_id,
                                                        one_minus_A_id, one_minus_A_transpose_id,
                                                        c_s)
        remoted_s = [bind_compute_s(chunk) for chunk in state_chunks]
        new_S = np.concatenate([ray.get(x) for x in remoted_s]).reshape((n_states1, n_states2))
        S[new_S >= 0] = new_S[new_S >= 0]
        if self_similarity:
            i_lower = np.tril_indices(len(S), -1)
            S[i_lower] = S.T[i_lower]

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
