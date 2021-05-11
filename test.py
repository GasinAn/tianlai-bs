import numpy
import h5py
import hdf5plugin

from dnb import reduce_precision


EXAMPLE = '/DATALINKS/D190611_3srcNP/3srcNP_20190622023112_20190622033112.hdf5'


def test():
    """Test for Tianlai data."""

    import os
    import time

    f = 0.01   # Precision reduction parameter.
    N = 244140 # Number of samples integrated (delta_f*delta_t).

    fsize = os.path.getsize(EXAMPLE)

    t_s = time.perf_counter()
    with h5py.File(EXAMPLE, 'r') as df:
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
