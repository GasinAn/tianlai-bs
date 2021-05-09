import numpy
import h5py
import hdf5plugin

from dnb import reduce_precision

def test():
    """Test reduce_precision for Tianlai data."""

    import os
    import time

    f = 0.01   # Precision reduction parameter.
    N = 100000 # Number of samples integrated (delta_f*delta_t).

    fsize = os.path.getsize('example.hdf5')

    t_s = time.perf_counter()
    with h5py.File('example.hdf5', 'r') as df:
        vis, blorder = df['vis'][...], df['blorder'][...]
    reduce_precision(vis, blorder, f/N)
    with h5py.File('example.bs.hdf5', 'w') as df:
        df.create_dataset('vis', data=vis, **hdf5plugin.Bitshuffle())
    t_e = time.perf_counter()

    print("Throughput(compress): %f MiB/s" %((fsize/1024**2)/(t_e-t_s)))

    del vis

    t_s = time.perf_counter()
    with h5py.File('example.bs.hdf5', 'r') as df:
        vis = df['vis'][...]
    t_e = time.perf_counter()

    print("Throughput(decompress): %f MiB/s" %((fsize/1024**2)/(t_e-t_s)))

    bs_fsize = os.path.getsize('example.bs.hdf5')
    print('Compression rate: %f %%' %((bs_fsize/fsize)*100))


if __name__=='__main__':
    test()
