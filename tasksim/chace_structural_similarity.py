import numpy as np
import ot
#from structural_similarity import structural_similarity as ss_2


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
    return delta_a[np.array(N_u)].swapaxes(0, 1)[np.array(N_v)].swapaxes(0, 1).min(axis=1).max()


def structural_similarity(action_dists1, action_dists2, reward_matrix1, reward_matrix2, out_neighbors_S1,
                          out_neighbors_S2, c_a=0.95, c_s=0.95, stop_rtol=1e-3,
                          stop_atol=1e-4, max_iters=1e5, zero_init=False):
    n_actions1, n_states1 = action_dists1.shape
    n_actions2, n_states2 = action_dists2.shape

    S = np.ones((n_states1, n_states2))
    A = np.ones((n_actions1, n_actions2))
    if zero_init:
        S = np.zeros((n_states1, n_states2))
        A = np.zeros((n_actions1, n_actions2))
        S = np.random.rand(n_states1, n_states2)
        A = np.random.rand(n_actions1, n_actions2)
    # S = np.eye(n_states1, n_states2)
    # A = np.eye(n_actions1, n_actions2)
    states1 = list(range(n_states1))
    states2 = list(range(n_states2))

    one_minus_c_a = 1 - c_a  # optimization

    last_S = S.copy()
    last_A = A.copy()

    done = False
    iter = 0
    while not done and iter < max_iters:
        for u in states1:
            for v in states2:
                # todo: is the end result symmetric? Can we skip about half the compute?
                # Note: Could we just use the P matrix to know what the out_neighbors are for actions and states?
                N_u = out_neighbors_S1[u]
                N_v = out_neighbors_S2[v]
                for alpha in N_u:
                    for beta in N_v:
                        p_alpha = action_dists1[alpha]
                        p_beta = action_dists2[beta]
                        d_rwd = abs(np.sum(reward_matrix1[alpha] * p_alpha) - np.sum(reward_matrix2[beta] * p_beta))
                        emd = ot.emd2(p_alpha, p_beta, 1 - S)  # Note: can be >1 by small rounding error
                        entry = 1 - one_minus_c_a * d_rwd - c_a * emd
                        A[alpha, beta] = entry
                        #A[beta, alpha] = entry  # have to keep track of whole matrix for directed hausdorff

        for u in states1:
            for v in states2:
                # TODO: figure out how to actually handle real absorbing states; mayhaps just augment?
                if len(out_neighbors_S1[u]) == 0 or len(out_neighbors_S2[v]) == 0:
                    continue  # skip
                #import pdb; pdb.set_trace()
                haus = max(directed_hausdorff_numpy(1 - A, out_neighbors_S1[u], out_neighbors_S2[v]),
                           directed_hausdorff_numpy((1 - A).T, out_neighbors_S2[v], out_neighbors_S1[u]))
                entry = c_s * (1 - haus)
                S[u, v] = entry
                #S[v, u] = entry  # Note: might be unnecessary, just need upper triangle of matrix due to symmetry

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

    out_neighbors_S2 = {0: [0, 1], 1: [2, 3], 2: [], 3: [], 4: [], 5: []}
    P2 = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.2, 0.5, 0.3],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.15, 0.15, 0.25, 0.45]])
    R2 = np.array([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.6, 0.7, 0.9],
                  [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.2, 0.4, 0.6, 0.5]])

    # out_neighbors_S2 = {0: [0, 1], 1: [2, 3], 2: [], 3: [], 4: [], 5: [], 6: []}
    # out_neighbors_S2 = {0: [0, 1], 1: [2, 3], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [0, 1], 8: [2, 3], 9: [], 10: [], 11: [], 12: [], 13: []}
    # P2 = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.1, 0.5, 0.3, 0.1],
    #               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.15, 0.15, 0.25, 0.45, 0.0]])
    # R2 = np.array([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.6, 0.7, 0.9, 0.8],
    #               [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.2, 0.4, 0.6, 0.5, 0.0]])

    c_s = 0.95
    c_a = 0.95
    s, a, num_iters, d = structural_similarity(P, P2, R, R2, out_neighbors_S, out_neighbors_S2, c_a, c_s)
    # s, a, num_iters, d = structural_similarity(P, P, R, R, out_neighbors_S, out_neighbors_S, c_a, c_s)
    # s, a, num_iters, d = ss_2(np.concatenate((P, P2), axis=1), np.concatenate((R, R2), axis=1), out_neighbors_S2, c_a, c_s)

    print("S\n", s)
    print("delta_S\n", 1 - s)
    print("A\n", a)
    print("num iters:", num_iters)
    print("converged:", d)

    ns, nt = s.shape
    a = np.array([1/ns for _ in range(ns)])
    b = np.array([1/nt for _ in range(nt)])
    print(ot.emd2(a, b, 1-s))

