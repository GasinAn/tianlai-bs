import os
import time

import numpy
import h5py
import hdf5plugin

from dnb import reduce_precision

with h5py.File('example.hdf5', 'r') as df:
    vis, blorder = df['vis'][...], df['blorder'][...]
    vis_size = vis.size

t_s = time.perf_counter()
vis_r, vis_i = reduce_precision(vis, blorder, 1e-5/100)
t_e = time.perf_counter()

rate = vis_size*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
print("Throughput(reduce_precision): %f MiB/s" %(rate/1024**2))

del vis

with h5py.File('dnb_example.hdf5', 'w') as f:
    t_s = time.perf_counter()
    f.create_dataset('vis_r', data=vis_r, **hdf5plugin.Bitshuffle())
    f.create_dataset('vis_i', data=vis_i, **hdf5plugin.Bitshuffle())
    t_e = time.perf_counter()

rate = vis_size*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
print("Throughput(bitshuffle_compress): %f MiB/s" %(rate/1024**2))

with h5py.File('dnb_example.hdf5', 'r') as f:
    t_s = time.perf_counter()
    vis_r_ = f['vis_r'][...]
    vis_i_ = f['vis_i'][...]
    t_e = time.perf_counter()

rate = vis_size*numpy.dtype(numpy.complex64).itemsize/(t_e-t_s)
print("Throughput(bitshuffle_decompress): %f MiB/s" %(rate/1024**2))

if numpy.any(vis_r_!=vis_r) or numpy.any(vis_i_!=vis_i):
    raise ValueError('Data changed after I/O.')

fsize = os.path.getsize('dnb_example.hdf5')
rate = fsize/(vis_size*numpy.dtype(numpy.complex64).itemsize)
print('Compression rate: %f %%' %(100*rate))
