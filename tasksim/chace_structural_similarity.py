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
                          stop_atol=1e-4, max_iters=1e5, zero_init=True):
    n_actions1, n_states1 = action_dists1.shape
    n_actions2, n_states2 = action_dists2.shape

    S = np.ones((n_states1, n_states2))
    A = np.ones((n_actions1, n_actions2))
    if zero_init:
        S = np.zeros((n_states1, n_states2))
        A = np.zeros((n_actions1, n_actions2))
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

        for u in states1:
            for v in states2:
                # TODO: figure out how to actually handle real absorbing states; mayhaps just augment?
                if len(out_neighbors_S1[u]) == 0 or len(out_neighbors_S2[v]) == 0:
                    if len(out_neighbors_S1[u]) == 0 and len(out_neighbors_S2[v]) == 0:
                        # absorbing to absorbing
                        # set haus distance to 0, i.e. they are equivalent
                        entry = c_s
                    else:
                        # absorbing to empty
                        # set haus distance to inf, or in this case, 1, i.e. they are maximally dissimilar
                        entry = 0
                else:
                    haus = max(directed_hausdorff_numpy(1 - A, out_neighbors_S1[u], out_neighbors_S2[v]),
                               directed_hausdorff_numpy((1 - A).T, out_neighbors_S2[v], out_neighbors_S1[u]))
                    entry = c_s * (1 - haus)
                S[u, v] = entry

        if np.allclose(A, last_A, rtol=stop_rtol, atol=stop_atol) and np.allclose(S, last_S, rtol=stop_rtol,
                                                                                  atol=stop_atol):
            # Note: Could update this to use specified more specific convergence criteria
            done = True
        else:
            last_S = S.copy()
            last_A = A.copy()

        iter += 1

    return S, A, iter - 1, done  # return done to report that it converged (or didn't)


def wang_structural_similarity(action_dists, reward_matrix, out_neighbors_S, c_a=0.95, c_s=0.95, stop_rtol=1e-5,
                          stop_atol=1e-8, max_iters=1e5):
    """
    Compute the structural similarity of an MDP graph as inspired by Wang et. al. paper:
        https://www.ijcai.org/Proceedings/2019/0511.pdf.
        Note: We still want to augment this to work for different numbers of states/actions for different MDPs
    :param action_dists: (np.array) P(s' | s, a) for each action node in the MDP graph.
    :param reward_matrix: (np.array) r(s, a) for each action node in the MDP graph.
    :param out_neighbors_S: (dict) Dictionary mapping the state nodes to their corresponding action nodes.
    :param c_a: (float) a constant in [0, 1] discounting the impact of the neighbors Nα and Nβ on the pair of state
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

    last_S = S.copy()
    last_A = A.copy()

    done = False
    iter = 0
    while not done and iter < max_iters:
        for u in states:
            for v in states[u + 1:]:
                # Note: Could we just use the P matrix to know what the out_neighbors are for actions and states?
                N_u = out_neighbors_S[u]
                N_v = out_neighbors_S[v]
                for alpha in N_u:
                    for beta in N_v:
                        p_alpha = action_dists[alpha]
                        p_beta = action_dists[beta]
                        d_rwd = abs(np.sum(reward_matrix[alpha] * p_alpha) - np.sum(reward_matrix[beta] * p_beta))
                        emd = ot.emd2(p_alpha, p_beta, 1 - S)  # Note: can be >1 by small rounding error
                        entry = 1 - one_minus_c_a * d_rwd - c_a * emd
                        A[alpha, beta] = entry
                        A[beta, alpha] = entry  # have to keep track of whole matrix for directed hausdorff

        for u in states:
            for v in states[u + 1:]:
                if len(out_neighbors_S[u]) == 0 or len(out_neighbors_S[v]) == 0:
                    continue  # skip
                haus = max(directed_hausdorff_numpy(1 - A, out_neighbors_S[u], out_neighbors_S[v]),
                           directed_hausdorff_numpy(1 - A, out_neighbors_S[v], out_neighbors_S[u]))
                entry = c_s * (1 - haus)
                S[u, v] = entry
                S[v, u] = entry  # Note: might be unnecessary, just need upper triangle of matrix due to symmetry

        print(iter + 1)
        print("S\n", S)
        print("A\n", A)
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
    np.set_printoptions(linewidth=200)
    # Example based on the MDP described in Figure 1 of https://www.ijcai.org/Proceedings/2019/0511.pdf.
    # NOTE: for cross_structural_similarity, requires all states to have at least one action
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
    import tasksim.structural_similarity as sim
    s, a, num_iters, d = structural_similarity(P, P2, R, R2, out_neighbors_S, out_neighbors_S2, c_a, c_s)
    s2, a2, num_iters2, d2 = sim.cross_structural_similarity(P, P2, R, R2, out_neighbors_S, out_neighbors_S2, c_a, c_s)
    # s, a, num_iters, d = structural_similarity(P, P, R, R, out_neighbors_S, out_neighbors_S, c_a, c_s)
    # s, a, num_iters, d = ss_2(np.concatenate((P, P2), axis=1), np.concatenate((R, R2), axis=1), out_neighbors_S2, c_a, c_s)

    print("S\n", s)
    print("S2\n", s2)
    print("delta_S\n", 1 - s)
    print("delta_S2\n", 1 - s2)
    print("A\n", a)
    print("A2\n", a2)
    print("num iters:", num_iters)
    print("num iters2:", num_iters2)
    print("converged:", d)
    print("converged2:", d2)

    ns, nt = s.shape
    a = np.array([1/ns for _ in range(ns)])
    b = np.array([1/nt for _ in range(nt)])
    print('Score', ot.emd2(a, b, 1-s))
    print('Score2 not normalized', sim.final_score(s2), c_a, c_s)
    print('Score2 normalized', sim.normalize_score(sim.final_score(s2), c_a, c_s))

