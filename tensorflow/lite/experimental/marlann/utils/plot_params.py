#! /usr/bin/env python
from getopt import getopt
import numpy as np
import sys
import matplotlib.pyplot as plt
import tflite_utils

from tflite.Model import Model

class model_summary:
  def __init__(self, fname):
    buf = bytearray(open(fname, 'rb').read())
    self.model = Model.GetRootAsModel(buf, 0)
    self.subgraph = self.model.Subgraphs(0)

  def dump_bufs(self):
    print('Model Description: ', self.model.Description())

    for ii in range(self.model.BuffersLength()):
      self.dump_buf(ii)

  def dump_buf(self, ii):
    tbuf = self.model.Buffers(ii).DataAsNumpy()
    if isinstance(tbuf, np.ndarray):
      print('Buffer shape: {}  type: {}'.format(tbuf.shape, tbuf.dtype))
    else:
      print('skip', ii)

  def dump_tensors(self):
    print('Model Description: ', self.model.Description())

    for ii in range(self.subgraph.TensorsLength()):
      tensor = self.subgraph.Tensors(ii)
      npbuf = tflite_utils.ConvertBuffer(self.model, tensor, True, False)
      print('Tensor {}: shape: {}  type: {} buffer: {}'.format(tensor.Name().decode('ascii'),
                                                               tensor.ShapeAsNumpy(),
                                                               tensor.Type(),
                                                               tensor.Buffer()))

      if npbuf is not None:
        plt.hist(npbuf.flatten(), 256)
        print('shape: {} type: {}  min,max: {} {}'.format(npbuf.shape,
                                                          npbuf.dtype,
                                                          np.min(npbuf.flatten()),
                                                          np.max(npbuf.flatten())
        ))
        self.dump_buf(tensor.Buffer())
        plt.show()

def main(argv):
  md = model_summary(argv[1])
  md.dump_tensors()

if __name__ == '__main__':
  main(sys.argv)
