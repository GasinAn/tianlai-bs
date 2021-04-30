import numpy as np

cimport numpy as np
cimport cython
from libc.math cimport sqrt

np.import_array()


DEF HEX_00400000 = 4194304
DEF HEX_7F800000 = 2139095040
DEF HEX_80000000 = -2147483648
DEF HEX_FF800000 = -8388608


@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_precision(
        np.ndarray[np.complex64_t, ndim=3, mode='c'] vis not None,
        np.ndarray[np.int32_t, ndim=2, mode='c'] blorder not None,
        np.float64_t g_factor,
        ):

    cdef np.int32_t ntime = vis.shape[0]
    cdef np.int32_t nfreq = vis.shape[1]
    cdef np.int32_t nprod = vis.shape[2]

    cdef np.int32_t nchan = blorder[:,0].max()

    cdef np.int32_t i
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] auto_inds
    auto_inds = np.empty(nchan, np.int32)
    for i in xrange(nprod):
        if blorder[i,0]==blorder[i,1]:
            auto_inds[blorder[i,0]-1] = i

    cdef np.int32_t n0, n1, n2

    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] auto_vis
    auto_vis = np.empty((ntime, nchan), np.float32)

    cdef np.int32_t blorder0, blorder1
    cdef np.int32_t is_auto

    cdef np.float32_t auto0, auto1

    cdef np.float64_t auto_g_factor = g_factor*12.0
    cdef np.float64_t corr_g_factor = g_factor*6.0

    cdef np.float32_t g_max

    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] vis_r
    cdef np.ndarray[np.float32_t, ndim=3, mode='c'] vis_i
    vis_r = np.empty((nfreq, nprod, ntime), np.float32)
    vis_i = np.empty((nfreq, nprod, ntime), np.float32)

    for n0 in xrange(nfreq):
        auto_vis[:,:] = vis[:,n0,auto_inds].real

        for n1 in xrange(nprod):
            blorder0 = blorder[n1,0]
            blorder1 = blorder[n1,1]
            is_auto = blorder0==blorder1

        for n2 in xrange(ntime):
                auto0 = auto_vis[n2,blorder0-1]
                auto1 = auto_vis[n2,blorder1-1]

                if is_auto:
                    g_max = <np.float32_t> sqrt(auto0*auto1*auto_g_factor)

                    vis_r[n0,n1,n2] = bit_round(vis[n2,n0,n1].real, g_max)
                    vis_i[n0,n1,n2] = 0

                else:
                    g_max = <np.float32_t> sqrt(auto0*auto1*corr_g_factor)

                    vis_r[n0,n1,n2] = bit_round(vis[n2,n0,n1].real, g_max)
                    vis_i[n0,n1,n2] = bit_round(vis[n2,n0,n1].imag, g_max)

    return vis_r, vis_i

def bit_round_py(np.float32_t val, np.float32_t g_max):
    """Python wrapper of C version, for testing."""
    return bit_round(val, g_max)

cdef inline np.float32_t bit_round(np.float32_t val, np.float32_t g_max):
    """
    Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).
    Warning: undefined behavior when delta_exponent >= 32.

    """

    cdef np.uint32_t *p_val = <np.uint32_t*> &val
    cdef np.uint32_t *p_g_max = <np.uint32_t*> &g_max

    cdef np.int32_t exponent_val = p_val[0] & HEX_7F800000
    cdef np.int32_t exponent_g_max = p_g_max[0] & HEX_7F800000

    cdef np.int32_t delta_exponent = (exponent_val - exponent_g_max) >> 23

    # Situation: delta_exponent >= 0,
    # return trunc_to_mul_of_2_to_b(val + sgn(val) * 2 ** (b - 1)).
    cdef np.uint32_t val_r_dexp_ge_0
    val_r_dexp_ge_0 = p_val[0] + (HEX_00400000 >> delta_exponent)
    val_r_dexp_ge_0 = val_r_dexp_ge_0 & (HEX_FF800000 >> delta_exponent)
    val_r_dexp_ge_0 = (delta_exponent > -1) * val_r_dexp_ge_0

    # Situation: delta_exponent == -1,
    # return sgn(val) * g_max.
    cdef np.uint32_t val_r_dexp_eq_m1
    val_r_dexp_eq_m1 = (p_val[0] & HEX_80000000) | exponent_g_max
    val_r_dexp_eq_m1 = (delta_exponent == -1) * val_r_dexp_eq_m1

    cdef np.uint32_t val_r = val_r_dexp_ge_0 + val_r_dexp_eq_m1
    return (<np.float32_t*> &val_r)[0]
