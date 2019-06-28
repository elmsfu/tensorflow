import numpy as np
from getopt import getopt
import sys

def read(fobj, dims, order, offset=0, nbytes=-1):
    # skip to offset and read bytes
    fobj.seek(offset)
    data = fobj.read(nbytes)

    # interpret to ndarray
    arr = np.ndarray(dims, dtype=np.uint8, buffer=data)

    # reorder to NHWC
    if order == 'CWHN':
        return arr.transpose()
    elif order == 'NHWC':
        return arr
    else:
        assert (), 'unsupported order'

def comp_print(*args):
    zz = zip(*[str(xx).split('\n') for xx in args])
    return ''.join(['\t'.join(xx) + '\n' for xx in zz])

def main(case, dims, offset):
    tf = read(open('test/test{:03d}_tflite.bin'.format(case), 'rb'),
              dims,
              order='NHWC')
    ml = read(open('test/test{:03d}_out.bin'.format(case), 'rb'),
              [xx for xx in reversed(dims)],
              order='CWHN',
              offset=offset,
              nbytes=len(tf.flatten()))

    tf2 = np.minimum(tf, 127)
    if (tf2-ml).any():
        print(comp_print(tf2, ml, tf2-ml))
        return 1

if __name__ == '__main__':
    opts, rem = getopt(sys.argv[1:], 'c:d:o:')

    for opt, arg in opts:
        if opt == '-c':
            case = int(arg, base=0)
        elif opt == '-d':
            dims = eval('[{}]'.format(arg))
        elif opt == '-o':
            offset = int(arg, base=0)
        else:
            assert (), 'unsupported option'

    ret = main(case, dims, offset)
    sys.exit(ret)
