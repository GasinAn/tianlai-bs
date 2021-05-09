"""
Tianlai Visibility Data Precision Reduction.

See arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015 for description of method.

Author: Jiachen An <Gasin185@163.com>
Website: https://github.com/GasinAn/tianlai-bs/blob/main/dnb.pyx

Copyright (c) 2021 Jiachen An (Gasin185@163.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

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
    """
    Reduce visibility precision for Tianlai data.

    Parameters
    ----------
    vis: array of complex64 with shape (ntime, nfreq, nprod)
        Visibilities to be processed.
    blorder: array of int32 with shape (nprod, 2)
        Baseline order (numbers of channels start from 1).
    f_N: float64
        f/N. Controls degree of precision reduction.
        f is the maximum fractional increase in noise.
        N is the number of samples entering the integrations.

    Returns
    -------
    None.
    After calling this function, vis will be the data after reducing precision.

    Notes
    -----
    It is assumed that the cross-correlations between channels are much more
    smaller than the auto-correlations.
    See docstring of function bit_round for limitations.
    See arXiv:1503.00638v3 [astro-ph.IM] 16 Sep 2015 for more details.

    """

    cdef np.int32_t ntime = vis.shape[0]
    cdef np.int32_t nfreq = vis.shape[1]
    cdef np.int32_t nprod = vis.shape[2]

    cdef np.int32_t nchan = blorder[:,0].max()

    cdef np.int32_t kt, kf, kp

    for kp in xrange(nprod):
        blorder[kp,0] = blorder[kp,0]-1
        blorder[kp,1] = blorder[kp,1]-1

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] auto_inds
        auto_inds = np.empty(nchan, np.int32)

    for kp in xrange(nprod):
        if blorder[kp,0]==blorder[kp,1]:
            auto_inds[blorder[kp,0]] = kp

    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] auto_vis
    auto_vis = np.empty(nchan, np.float32)

    cdef np.int32_t blorder0, blorder1
    cdef np.float32_t auto0, auto1

    cdef np.float64_t auto_g_factor = g_factor*12.0
    cdef np.float64_t corr_g_factor = g_factor*6.0
    cdef np.float32_t g_max

    for kt in xrange(ntime):
        for kf in xrange(nfreq):
            auto_vis[:] = vis[kt,kf,auto_inds].real
            for kp in xrange(nprod):
                blorder0 = blorder[kp,0]
                blorder1 = blorder[kp,1]
                auto0 = auto_vis[blorder0]
                auto1 = auto_vis[blorder1]
                if blorder0==blorder1:
                    g_max = <np.float32_t> sqrt(auto0*auto1*auto_g_factor)
                    vis[kt,kf,kp].real = bit_round(vis[kt,kf,kp].real, g_max)
                else:
                    g_max = <np.float32_t> sqrt(auto0*auto1*corr_g_factor)
                    vis[kt,kf,kp].real = bit_round(vis[kt,kf,kp].real, g_max)
                    vis[kt,kf,kp].imag = bit_round(vis[kt,kf,kp].imag, g_max)

    for kp in xrange(nprod):
        blorder[kp,0] = blorder[kp,0]+1
        blorder[kp,1] = blorder[kp,1]+1

def bit_round_py(np.float32_t val, np.float32_t g_max):
    """Python wrapper of C version, for testing."""
    return bit_round(val, g_max)

cdef inline np.float32_t bit_round(np.float32_t val, np.float32_t g_max):
    """
    Round val to val_r = n*2**b (int n; int b = max(b: 2**b <= g_max)).
    Warning: undefined behavior when delta_exponent >= 32.
    Warning: if val or g_max is NAN, the output may not be NAN.

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
    # return sgn(val) * 2 ** b.
    cdef np.uint32_t val_r_dexp_eq_m1
    val_r_dexp_eq_m1 = (p_val[0] & HEX_80000000) | exponent_g_max
    val_r_dexp_eq_m1 = (delta_exponent == -1) * val_r_dexp_eq_m1

    cdef np.uint32_t val_r = val_r_dexp_ge_0 + val_r_dexp_eq_m1
    return (<np.float32_t*> &val_r)[0]
