import cython
cimport cython
cimport numpy as np
import numpy as np

from cython.parallel import prange

ctypedef np.float32_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def outer_mul(DTYPE_t[:] a, DTYPE_t[:] b, DTYPE_t[:,:] out):
    # cdef int m = a.shape[0]
    # cdef int n = b.shape[0]
    cdef int m = len(a)
    cdef int n = len(b)
    cdef int i
    cdef int j
    #cdef DTYPE_t[:,:] result
    #result = out
    with nogil:
        for i in prange(m):
            for j in range(n):
                #result[i, j] = a[i]*b[j]
                out[i, j] = a[i]*b[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_add(DTYPE_t[:,:] A, DTYPE_t[:,:] B, int shape_0, int shape_1):
    cdef int i
    cdef int j
    with nogil:
        for i in prange(shape_0):
            for j in range(shape_1):
                A[i,j] += B[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_perturbed_A(int n_eigen,
                          DTYPE_t[:,:] x_k,
                          DTYPE_t[:] lambda_k,
                          DTYPE_t[:] delta_lambda_k,
                          long[:] target,
                          long N_nodes
                         ):
    cdef int i
    cdef int j #len target
    perturbed_A = np.zeros((len(target), N_nodes), dtype=np.float32)
    cdef DTYPE_t[:,:] perturbed_A_view = perturbed_A
    cdef DTYPE_t[:,:] outer_prod
    cdef DTYPE_t[:] x_k_target
    cdef DTYPE_t[:,:] outer_prod_view
    cdef DTYPE_t scalar
    tmp_out = np.empty(((len(target), N_nodes)), dtype=np.float32)

    for i in range(n_eigen):
        x_k_target = np.take(x_k[:,i],target)
        scalar = (lambda_k[i] + delta_lambda_k[i])
        for j in range(len(target)):
            x_k_target[j] = x_k_target[j] * scalar

        outer_mul(x_k_target, x_k[:,i], tmp_out)
        matrix_add(perturbed_A_view, tmp_out, len(target), N_nodes)

    return perturbed_A

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_perturbed_B(int n_sing_values,
                          DTYPE_t[:,:] u,
                          DTYPE_t[:] s,
                          DTYPE_t[:,:] v,
                          DTYPE_t[:] delta_svs,
                          long[:] target):
    cdef int i
    cdef int j
    perturbed_B = np.zeros((len(target), v.shape[1]), dtype=np.float32)
    cdef DTYPE_t[:,:] perturbed_B_view = perturbed_B
    cdef DTYPE_t[:,:] outer_prod
    cdef DTYPE_t[:] u_target
    cdef DTYPE_t scalar
    tmp_out = np.empty((len(target), v.shape[1]), dtype=np.float32)

    for i in range(n_sing_values):
        u_target = np.take(u[:,i], target)
        scalar = s[i] + delta_svs[i]

        for j in range(len(target)):
            u_target[j] = u_target[j] * scalar

        outer_mul(u_target, v[i], tmp_out)
        matrix_add(perturbed_B_view, tmp_out, len(target), v.shape[1])

    return perturbed_B

