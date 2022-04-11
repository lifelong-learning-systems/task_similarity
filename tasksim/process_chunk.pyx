import numpy as np
import ot
import ot.lp

#np.import_array()
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free

cdef extern from "EMD.h":
    int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, int maxIter) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_chunk(np.ndarray[DTYPE_t, ndim=3] chunk,
                  np.ndarray[DTYPE_t, ndim=2] reward_diffs,
                  np.ndarray[DTYPE_t, ndim=2] action_dists1,
                  np.ndarray[DTYPE_t, ndim=2] action_dists2,
                  np.ndarray[DTYPE_t, ndim=2] one_minus_S,
                  DTYPE_t c_a,
                  DTYPE_t emd_maxiters):
    cdef int n_a1 = chunk.shape[0]
    cdef int n_a2 = chunk.shape[1]
    cdef int count = -1
    cdef np.ndarray[DTYPE_t, ndim=1] entries = np.zeros([1, n_a1*n_a2], dtype=DTYPE).squeeze() * np.nan
    cdef np.ndarray[DTYPE_t, ndim=1] x, y
    cdef DTYPE_t d_rwd, d_emd, entry
    cdef int i, j, alpha, beta

    for i in range(n_a1):
        for j in range(n_a2):
            count += 1
            if np.isnan(chunk[i, j][0]) or np.isnan(chunk[i, j][1]):
                entries[count] = np.nan
                continue
            alpha = int(chunk[i, j][0])
            beta = int(chunk[i, j][1])
            d_rwd = reward_diffs[alpha, beta]
            x = action_dists1[alpha, :]
            y = action_dists2[beta, :]
            d_emd = ot.lp.emd_c(x, y, one_minus_S, emd_maxiters)[1]
            entry = 1 - (1 - c_a) * d_rwd - c_a * d_emd
            entries[count] = entry

    return entries

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api double emd_c_pure(double* a, double* b, double* M, int n1, int n2, int max_iter):
    cdef double cost = 0
    cdef double* alpha
    cdef double* beta
    cdef double* G
    cdef int i,j,k
    cdef int result_code

    alpha = <double*> malloc(n1*cython.sizeof(double))
    beta = <double*> malloc(n2*cython.sizeof(double))
    G = <double*> malloc(n1*n2*cython.sizeof(double))

    for i in range(n1):
        alpha[i] = 0
    for j in range(n2):
        beta[j] = 0
    for k in range(n1*n2):
        G[k] = 0

    result_code = EMD_wrap(n1, n2, <double*> a, <double*> b, <double*> M, <double*> G, <double*> alpha, <double*> beta, <double*> &cost, max_iter)

    free(alpha)
    free(beta)
    free(G)

    return cost

@cython.boundscheck(False)
@cython.wraparound(False)
cdef api emd_c_pure_chunk(int chunk_n1, int chunk_n2,
                          int num_actions1, int num_actions2,
                          int num_states1, int num_states2,
                          double* chunk, double* reward_diffs,
                          double* actions1, double* actions2,
                          double* one_minus_S,
                          double* entries,
                          double c_a, int emd_maxiters):
    cdef double entry
    cdef int count = -1
    cdef int i, j, k, idx, alpha, beta
    cdef double d_rwd, d_emd
    cdef double* x
    cdef double* y

    #chunks = chunk_n1 x chunk_n2
    #reward_diffs = num_actions1 x num_actions2
    #actions1 = num_actions1 x num_states1
    #actions2 = num_actions2 x num_states2
    #one_minus_S = num_states1 x num_states2

    for i in range(chunk_n1*chunk_n2):
        entries[i] = -1

    for j in range(chunk_n1):
        for k in range(chunk_n2):
            count += 1
            #indexing 3d is harder than I thought...
            idx = 0 + 2*(k + chunk_n2*j)
            alpha = int(chunk[idx])
            beta = int(chunk[idx + 1])
            if alpha < 0 or beta < 0:
                continue
            d_rwd = reward_diffs[num_actions2*alpha + beta]
            x = &actions1[num_states1*alpha]
            y = &actions2[num_states2*beta]
            d_emd = emd_c_pure(<double*>x, <double*>y, <double*>one_minus_S, num_states1, num_states2, emd_maxiters)
            entry = (1 - c_a) * d_rwd + c_a * d_emd
            entries[count] = entry
