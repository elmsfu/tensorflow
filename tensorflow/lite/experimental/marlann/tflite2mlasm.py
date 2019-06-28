#! /usr/bin/env python
import collections
from getopt import getopt
import importlib
from io import StringIO
import numpy as np
from os.path import basename
import sys

import tflite_utils

from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.FullyConnectedOptionsWeightsFormat import FullyConnectedOptionsWeightsFormat
from tflite.Padding import Padding


class TFLiteConverter:
  __slots__ = ['model', 'subgraph']

  def __init__(self, fname):
    buf = bytearray(open(fname, 'rb').read())
    self.model = Model.GetRootAsModel(buf, 0)

    print('Model Description: ', self.model.Description())

    # check only 1 subgraph
    assert(1 == self.model.SubgraphsLength())
    self.subgraph = self.model.Subgraphs(0)
    # check only 1 input and one output
    assert(1 == self.subgraph.InputsLength())
    assert(1 == self.subgraph.OutputsLength())


  def traceOps(self):
    """trace from the output back to undefine inputs
    return a list of op indices in an order that when computed will compute the final output
    """
    tocomp_tensors = set()
    known_tensors = set(self.subgraph.InputsAsNumpy().tolist())
    for ii in range(self.subgraph.TensorsLength()):
      tensor = self.subgraph.Tensors(ii)
      bi = tensor.Buffer()
      if 0 == self.model.Buffers(bi).DataLength():
        tocomp_tensors.add(ii)
      else:
        known_tensors.add(ii)

    numOps = self.subgraph.OperatorsLength()

    def findOpsWithOutput(output):
      return [ii for ii in range(numOps) if output in self.subgraph.Operators(ii).OutputsAsNumpy()]

    outputs = self.subgraph.OutputsAsNumpy().tolist()
    op_order = []
    visited = []
    while len(outputs) > 0:
      output = outputs.pop(0)
      assert (visited.count(output) < 2), 'likely uncomputable node'
      visited.append(output)
      ops = findOpsWithOutput(output)

      for ii in ops:
        inputs = set(self.subgraph.Operators(ii).InputsAsNumpy())
        inputs.difference_update(known_tensors)
        if len(inputs) == 0:
          op_order.insert(0, ii)
          known_tensors.add(output)
        else:
          outputs = list(inputs) + [output] + outputs

    return op_order


class TfLiteAsmConverter(TFLiteConverter):
  __slots__ = ['code', 'kernel', 'sym_addr', 'data_buffers', 'buffers']

  PACKED = 'PACKED'
  PADDED = 'PADDED'

  store_op_map = {
    ActivationFunctionType.NONE: 'Store',
    ActivationFunctionType.RELU: 'ReLU',
  }

  layer = '''
// Load kernel to address 0 in code memory
{layer_label}:
LoadCode {kernel}, 0
ContinueLoad {kernel_end}/4 - {kernel}/4 - 1

// Load all coefficients
LoadCoeff0 {coef0_label}, 0
ContinueLoad {coef0_len} - 1

LoadCoeff1 {coef1_label}, 0
ContinueLoad {coef1_len} - 1

// Set pointers
SetLBP {bias_label}
SetVBP {input_label}
SetSBP {output_label}
SetCBP 0

Call {calc_label}
Return

'''

  def __init__(self, fname):
    super().__init__(fname)

    self.buffers = collections.OrderedDict()
    self.data_buffers = StringIO()
    self.kernel = StringIO()
    self.code = StringIO()
    # TODO: Generate start dynamically?
    self.sym_addr = 0x10000


  def GenDataBuffer(self, name, array):
    self.data_buffers.write('{}:\n'.format(name))
    self.data_buffers.write('// dims {}:\n'.format(array.shape))
    tbytes = array.tobytes()
    shape = array.shape
    step = shape[-1]
    fmt = step*'0x{:02x} ' + '\n'
    for ii in range(0, len(tbytes), step):
      self.data_buffers.write(fmt.format(*[int(xx) for xx in tbytes[ii:ii+step]]))
      if len(shape)==1 or (ii//step + 1)%shape[-2] == 0:
        self.data_buffers.write('\n')

  def AllocateBuffer(self, tensor):
    nbytes = np.prod(tensor.ShapeAsNumpy()) * np.dtype(tflite_utils.TF2NP[tensor.Type()]).itemsize
    name = tensor.Name().decode('ascii')
    self.buffers[name] = nbytes

  def WriteBuffers(self, io_buf):
    for name, nbytes in self.buffers.items():
      io_buf.write('.sym {} 0x{:x}\n'.format(name, self.sym_addr))
      self.sym_addr += nbytes

  def GenOutput(self, layers):
    buf_asm = StringIO()
    self.WriteBuffers(buf_asm)
    buf_asm.write('\n')

    buf_asm.write(self.Prelude(layers))

    buf_asm.write(self.code.getvalue())
    buf_asm.write('\n')

    buf_asm.write(self.kernel.getvalue())
    buf_asm.write('\n')

    buf_asm.write('\n.data\n')
    buf_asm.write(self.data_buffers.getvalue())
    buf_asm.write('\n')

    return buf_asm

  def Prelude(self, layers):
    calls = '\n'.join(['Call {}'.format(layer) for layer in layers])

    prelude = '''
.code 0x00000

Sync
{calls}
Return
'''
    return prelude.format(calls=calls)

  def layer_code(self, **kwargs):
    #assert (int(kwargs['coef0_len']) < 512)
    # assert (int(kwargs['coef1_len']) < 512)
    return self.layer.format(**kwargs)

  def SOFTMAX_gen(self, op, opts, opind, output_fmt):
    print('Skipping SOFTMAX layer')
    return None, output_fmt

  def RESHAPE_gen(self, op, opts, opind, output_fmt):
    print('Skipping RESPHAPE layer')
    return None, output_fmt

  def FULLY_CONNECTED_gen(self, op, opts, opind, output_fmt):
    # TODO: support more (at least NONE)
    assert (opts.FusedActivationFunction() in self.store_op_map.keys())
    assert (opts.WeightsFormat() == FullyConnectedOptionsWeightsFormat.DEFAULT), 'Only default weight format is supported'

    store_op = self.store_op_map[opts.FusedActivationFunction()]
    input_tensor = self.subgraph.Tensors(op.Inputs(0))
    filter_tensor = self.subgraph.Tensors(op.Inputs(1))
    bias_tensor = self.subgraph.Tensors(op.Inputs(2))
    output_tensor = self.subgraph.Tensors(op.Outputs(0))

    input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor)
    filter_buf = tflite_utils.ConvertBuffer(self.model, filter_tensor, False, False)
    bias_buf = tflite_utils.ConvertBuffer(self.model, bias_tensor)

    #input_depth, input_width, input_height = input_buf.shape
    filter_depth, filter_height = filter_buf.shape
    _, output_depth = output_tensor.ShapeAsNumpy()
    _, input_height, input_width, input_depth = input_tensor.ShapeAsNumpy()

    #assert (len(input_buf.shape) == 3)
    assert (len(filter_buf.shape) == 2)
    assert (len(bias_buf.shape) == 1)

    layer_label = 'fc_code_{}'.format(opind)
    kernel_label = 'fc_{}'.format(opind)
    kernel_end_label = kernel_label + '_end'
    calc_d_label = 'calc_fcd_{}'.format(opind)

    input_label = input_tensor.Name().decode('ascii')
    output_label = output_tensor.Name().decode('ascii')
    bias_label = bias_tensor.Name().decode('ascii')
    coef0_label = filter_tensor.Name().decode('ascii') + '_0'
    coef1_label = filter_tensor.Name().decode('ascii') + '_1'

    # split even and odd filters and pad to match input
    # reorder filter_buf to input shape byte order
    f2 = filter_buf.reshape([filter_depth, input_height, input_width, input_depth])
    f2 = f2.transpose([0, 3, 2, 1])
    f2 = np.concatenate( (f2.reshape([filter_depth, filter_height]),
                          np.zeros( [filter_depth, (8-filter_height)%8], dtype= f2.dtype)),
                         axis=1)

    coef0 = f2[::2,:]
    coef1 = f2[1::2,:]
    coef0_len = str((np.prod(coef0.shape) + 7)//8)
    coef1_len = str((np.prod(coef1.shape) + 7)//8)

    # generate labels
    self.GenDataBuffer(bias_label,  bias_buf)
    self.GenDataBuffer(coef0_label, coef0)
    self.GenDataBuffer(coef1_label, coef1)

    # Generate kernel code for compute module
    self.kernel.write('{}:\n'.format(kernel_label))

    # TODO: calulate shift from quantization parameters
    shift = 0
    # round up to multiple of 8
    filter_its = (filter_height+7) // 8
    self.kernel.write('LdSet 0\n')
    for h_ind in range(filter_its):
      self.kernel.write('MACC {}*8, {}\n'.format(h_ind, h_ind))
    self.kernel.write('{} {}, {}\n\n'.format(store_op, 0, shift))
    self.kernel.write('{}:\n'.format(kernel_end_label))

    # sequencer code
    # TODO: maybe change to take in the tensors
    self.code.write(self.layer_code(layer_label=layer_label,
                                    calc_label=calc_d_label,
                                    kernel=kernel_label, kernel_end=kernel_end_label,
                                    coef0_label=coef0_label, coef0_len=coef0_len,
                                    coef1_label=coef1_label, coef1_len=coef1_len,
                                    bias_label=bias_label,
                                    input_label=input_label,
                                    output_label=output_label))
    self.AllocateBuffer(output_tensor)

    self.code.write('{}:\n'.format(calc_d_label))
    exec_its = (output_depth + 1) // 2
    for out in range(exec_its):
      self.code.write('Execute {}, {}/4 - {}/4\n'.format(0, kernel_end_label, kernel_label))
      self.code.write('AddSBP {}\n'.format(2))
      self.code.write('AddCBP {}\n'.format(filter_its))
      self.code.write('AddLBP {}\n'.format(8))
    self.code.write('Return\n\n')

    return layer_label, self.PACKED

  def MAX_POOL_2D_gen(self, op, opts, opind, output_fmt):
    stride_h = opts.StrideH()
    stride_w = opts.StrideW()

    # TODO: support SAME padding
    assert (opts.Padding() == Padding.VALID), 'Only VALID padding supported at this time'

    filter_width = opts.FilterWidth()
    filter_height = opts.FilterHeight()

    assert (opts.FusedActivationFunction() in self.store_op_map.keys())
    store_op  = self.store_op_map[opts.FusedActivationFunction()]


    # dimensions
    assert (op.InputsLength() == 1)
    assert (op.OutputsLength() == 1)

    input_tensor = self.subgraph.Tensors(op.Inputs(0))
    output_tensor = self.subgraph.Tensors(op.Outputs(0))

    input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor)

    _, input_height, input_width, input_depth = input_tensor.ShapeAsNumpy()
    _, output_height, output_width, output_depth = output_tensor.ShapeAsNumpy()

    # Make mod 8
    output_height_pad = 8*((output_height+7)//8)
    input_height_pad =  8*((input_height+7)//8)

    layer_label = 'conv2d_code_{}'.format(opind)
    calc_w_label = 'calcw_{}'.format(opind)
    kernel_label = 'max_pool_{}'.format(opind)
    kernel_end_label = kernel_label + '_end'
    calc_z_label = 'calcz_{}'.format(opind)
    input_label = input_tensor.Name().decode('ascii')
    output_label = output_tensor.Name().decode('ascii')
    coef0_label = 'max_pool_c_{}'.format(opind) + '_0'
    coef1_label = 'max_pool_c_{}'.format(opind) + '_1'

    pad_out = (8 - filter_height) % 8
    even_weights = filter_height*[1] + pad_out*[0]
    odd_weights = [0] + filter_height*[1] + (pad_out - 1)*[0]
    coefs0 = np.array(even_weights + odd_weights, dtype=np.int8)
    coefs1 = np.array(0*coefs0, dtype=np.int8)

    coef0_len = str((len(coefs0)+7)//8)
    coef1_len = str((len(coefs1)+7)//8)

    self.GenDataBuffer(coef0_label, coefs0)
    self.GenDataBuffer(coef1_label, coefs1)

    # Generate kernel code for compute module
    self.kernel.write('{}:\n'.format(kernel_label))

    # TODO: calulate shift from quantization parameters
    shift = 0
    for out_y in range(0, output_height):
      if out_y%2 == 0:
        store_n = '0'
        coeff_offset = 0
      else:
        store_n = '0'
        coeff_offset = (len(even_weights) + 7)//8

      y_off = 2*(out_y//2)
      maxop = 'MMAXZ'
      for h_ind in range(filter_width):
        for ii in range(0, (filter_height+7)//8):
          self.kernel.write('{} {}*{}+{}, {}\n'.format(maxop, h_ind, input_height_pad, y_off, coeff_offset))
          maxop = 'MMAX'
      self.kernel.write('{}{} {}, {}\n\n'.format(store_op, store_n, out_y, shift))
    self.kernel.write('{}:\n'.format(kernel_end_label))

    # sequencer code
    # TODO: maybe change to take in the tensors
    self.code.write(self.layer_code(layer_label=layer_label,
                                    calc_label=calc_z_label,
                                    kernel=kernel_label, kernel_end=kernel_end_label,
                                    coef0_label=coef0_label, coef0_len=coef0_len,
                                    coef1_label=coef1_label, coef1_len=coef1_len,
                                    bias_label=0,
                                    input_label=input_label,
                                    output_label=output_label))
    self.AllocateBuffer(output_tensor)

    # compute 2ker column = kernel
    # compute 2ker conv = compute column, increment VBP, SBP
    # compute depth = compute conv, increment LBP, CBP, Set VBP, increment SBP
    self.code.write('{}:\n'.format(calc_w_label))
    exec_its = output_width
    VBPinc = input_height_pad
    if output_fmt == self.PADDED:
      SBPinc = output_height_pad
    elif output_fmt == self.PACKED:
      SBPinc = output_height
    for out_x in range(exec_its):
      self.code.write('Execute {}, {}/4 - {}/4\n'.format(0, kernel_end_label, kernel_label))
      self.code.write('AddVBP {}\n'.format(VBPinc))
      self.code.write('AddSBP {}\n'.format(SBPinc))
    self.code.write('Return\n\n')

    # output is stored kernel interleaved, This means it can't easily be used by
    # the next computtion which is likely expecting height interleaved.
    # - use RELU0 and RELU1 to store results in arbitrary location
    # - option to make coeff1 always shifted by stride_h of the kernels, biases
    #   would also need to be duplicated
    dec_len = -output_width * input_height_pad
    inc_len = (input_width - output_width) * input_height_pad
    self.code.write('{}:\n'.format(calc_z_label))
    for ii in range(output_depth):
      self.code.write('Call {}\n'.format(calc_w_label))

    self.code.write('Return\n\n')

    return layer_label, self.PADDED

  def CONV_2D_gen(self, op, opts, opind, output_fmt):
    assert (isinstance(opts, Conv2DOptions))

    # XXX: this is identical except the depthmultiplier to depthwise

    # TODO: support other Dilation Factors
    assert (opts.DilationHFactor() == 1 and opts.DilationWFactor() == 1)

    stride_h = opts.StrideH()
    stride_w = opts.StrideW()

    # TODO: support SAME padding
    assert (opts.Padding() == Padding.VALID), 'Only VALID padding supported at this time'

    assert (opts.FusedActivationFunction() in self.store_op_map.keys())
    store_op  = self.store_op_map[opts.FusedActivationFunction()]

    # XXX: end of duplicate code
    # dimensions
    assert (op.InputsLength() == 3)
    assert (op.OutputsLength() == 1)

    input_tensor = self.subgraph.Tensors(op.Inputs(0))
    filter_tensor = self.subgraph.Tensors(op.Inputs(1))
    bias_tensor = self.subgraph.Tensors(op.Inputs(2))
    output_tensor = self.subgraph.Tensors(op.Outputs(0))

    input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor)
    filter_buf = tflite_utils.ConvertBuffer(self.model, filter_tensor, transpose=[0,3,2,1], bSqueeze=False)
    bias_buf = tflite_utils.ConvertBuffer(self.model, bias_tensor)

    _, input_height, input_width, input_depth = input_tensor.ShapeAsNumpy()
    _, output_height, output_width, output_depth = output_tensor.ShapeAsNumpy()

    # Make mod 8
    output_height_pad = 8*((output_height+7)//8)

    assert (len(filter_buf.shape) == 4)
    assert (len(bias_buf.shape) == 1)

    filter_nouts, filter_depth, filter_width, filter_height = filter_buf.shape

    layer_label = 'conv2d_code_{}'.format(opind)
    calc_w_label = 'calcw_{}'.format(opind)
    kernel_label = 'conv2d_{}'.format(opind)
    kernel_end_label = kernel_label + '_end'
    calc_z_label = 'calcz_{}'.format(opind)
    input_label = input_tensor.Name().decode('ascii')
    output_label = output_tensor.Name().decode('ascii')
    bias_label = bias_tensor.Name().decode('ascii')
    coef0_label = filter_tensor.Name().decode('ascii') + '_0'
    coef1_label = filter_tensor.Name().decode('ascii') + '_1'
    coef0_len = str(filter_width * filter_height * filter_depth)
    coef1_len = str(filter_width * filter_height * filter_depth)

    bias_buf = bias_buf.repeat(2)
    filter_shift = np.concatenate( (np.zeros( (filter_nouts, filter_depth, filter_width, stride_h),
                                              dtype=filter_buf.dtype),
                                    filter_buf[:,:,:,:-stride_h]),
                                   axis=3)
    input_height = 8*((input_height + 7)//8)
    assert (input_height%8 == 0)

    self.GenDataBuffer(bias_label,  bias_buf)
    self.GenDataBuffer(coef0_label, filter_buf)
    self.GenDataBuffer(coef1_label, filter_shift)

    # Generate kernel code for compute module
    self.kernel.write('{}:\n'.format(kernel_label))

    # TODO: calulate shift from quantization parameters
    shift = 0
    for out_y in range(0, output_height, 2):
      w_rnd2 = 2*stride_w * (out_y // (2 * stride_w))
      if out_y == output_height-1:
        store_n = '0'
      else:
        store_n = ''
      # load bias into accumulators
      self.kernel.write('LdSet 0\n')
      for d_ind in range(filter_depth):
        for w_ind in range(filter_width):
          for ii in range(0, filter_height, 8):
            self.kernel.write('MACC {}*{}+{}+{}+{}, {}+{}+{}\n'.format(w_ind, input_height, w_rnd2, ii, d_ind* input_height*input_width,
                                                                       w_ind*(filter_height//8), ii, (d_ind*filter_height*filter_width)//8))
      self.kernel.write('{}{} {}, {}\n\n'.format(store_op, store_n, out_y, shift))
    self.kernel.write('{}:\n'.format(kernel_end_label))

    # sequencer code
    # TODO: maybe change to take in the tensors
    self.code.write(self.layer_code(layer_label=layer_label,
                                    calc_label=calc_z_label,
                                    kernel=kernel_label, kernel_end=kernel_end_label,
                                    coef0_label=coef0_label, coef0_len=coef0_len,
                                    coef1_label=coef1_label, coef1_len=coef1_len,
                                    bias_label=bias_label,
                                    input_label=input_label,
                                    output_label=output_label))
    self.AllocateBuffer(output_tensor)

    # compute 2ker column = kernel
    # compute 2ker conv = compute column, increment VBP, SBP
    # compute depth = compute conv, increment LBP, CBP, Set VBP, increment SBP
    self.code.write('{}:\n'.format(calc_w_label))
    exec_its = output_width
    VBPinc = input_height
    if output_fmt == self.PADDED:
      SBPinc = output_height_pad
    elif output_fmt == self.PACKED:
      SBPinc = output_height
    for out_x in range(exec_its):
      self.code.write('Execute {}, {}/4 - {}/4\n'.format(0, kernel_end_label, kernel_label))
      self.code.write('AddVBP {}\n'.format(VBPinc))
      self.code.write('AddSBP {}\n'.format(SBPinc))
    self.code.write('Return\n\n')

    # output is stored kernel interleaved, This means it can't easily be used by
    # the next computtion which is likely expecting height interleaved.
    # - use RELU0 and RELU1 to store results in arbitrary location
    # - option to make coeff1 always shifted by stride_h of the kernels, biases
    #   would also need to be duplicated
    dec_len = -output_width * input_height
    inc_len = (input_width - output_width) * input_height
    self.code.write('{}:\n'.format(calc_z_label))
    for ii in range(output_depth):
      self.code.write('Call {}\n'.format(calc_w_label))
      # reset to start of depth, increment
      self.code.write('AddVBP {}\n'.format(dec_len))
      # move to next kernel and next set of biases
      self.code.write('AddCBP {}\n'.format((filter_height*filter_depth*filter_width)//8))
      self.code.write('AddLBP {}\n'.format(8))

    self.code.write('Return\n\n')

    return layer_label, self.PADDED

  def DEPTHWISE_CONV_2D_gen(self, op, opts, opind, output_fmt):
    assert (isinstance(opts, DepthwiseConv2DOptions))

    # TODO: support other Dilation Factors
    assert (opts.DilationHFactor() == 1 and opts.DilationWFactor() == 1)

    depth_mult = opts.DepthMultiplier()
    stride_h = opts.StrideH()
    stride_w = opts.StrideW()

    # TODO: support SAME padding
    assert (opts.Padding() == Padding.VALID), 'Only VALID padding supported at this time'

    assert (opts.FusedActivationFunction() in self.store_op_map.keys())
    store_op  = self.store_op_map[opts.FusedActivationFunction()]

    # dimensions
    assert (op.InputsLength() == 3)
    assert (op.OutputsLength() == 1)

    input_tensor = self.subgraph.Tensors(op.Inputs(0))
    filter_tensor = self.subgraph.Tensors(op.Inputs(1))
    bias_tensor = self.subgraph.Tensors(op.Inputs(2))
    output_tensor = self.subgraph.Tensors(op.Outputs(0))

    input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor)
    filter_buf = tflite_utils.ConvertBuffer(self.model, filter_tensor)
    bias_buf = tflite_utils.ConvertBuffer(self.model, bias_tensor)

    _, input_height, input_width, input_depth = input_tensor.ShapeAsNumpy()
    _, output_height, output_width, output_depth = output_tensor.ShapeAsNumpy()

    # Make mod 8
    output_height_pad = 8*((output_height+7)//8)

    #assert (len(input_buf.shape) == 3)
    assert (len(filter_buf.shape) == 3)
    assert (len(bias_buf.shape) == 1)
    # assert (len(output_buf.shape) == 4)

    filter_depth, filter_width, filter_height = filter_buf.shape

    layer_label = 'depthwise_code_{}'.format(opind)
    calc_w_label = 'calcw_{}'.format(opind)
    kernel_label = 'depthwise_{}'.format(opind)
    kernel_end_label = kernel_label + '_end'
    calc_z_label = 'calcz_{}'.format(opind)
    input_label = input_tensor.Name().decode('ascii')
    output_label = output_tensor.Name().decode('ascii')
    bias_label = bias_tensor.Name().decode('ascii')
    coef0_label = filter_tensor.Name().decode('ascii') + '_0'
    coef1_label = filter_tensor.Name().decode('ascii') + '_1'
    coef0_len = str(filter_width * filter_depth)
    coef1_len = str(filter_width * filter_depth)

    bias_buf = bias_buf.repeat(2)
    filter_shift = np.concatenate( (np.zeros( (filter_depth, filter_width, stride_h),
                                              dtype=filter_buf.dtype),
                                    filter_buf[:,:,:-stride_h]),
                                   axis=2)
    # generate labels
    # TODO: input should be generated by the privous layer or top level
    #self.GenDataBuffer(input_label,  input_buf)
    # XXX:

    input_height = 8*((input_height + 7)//8)
    assert (input_height%8 == 0)

    self.GenDataBuffer(bias_label,  bias_buf)
    self.GenDataBuffer(coef0_label, filter_buf)
    self.GenDataBuffer(coef1_label, filter_shift)

    # Generate kernel code for compute module
    self.kernel.write('{}:\n'.format(kernel_label))

    # TODO: calulate shift from quantization parameters
    shift = 0
    for out_y in range(0, output_height, 2):
      w_rnd2 = 2*stride_w * (out_y // (2 * stride_w))
      if out_y == output_height-1:
        store_n = '0'
      else:
        store_n = ''
      # load bias into accumulators
      self.kernel.write('LdSet 0\n')
      for h_ind in range(filter_width):
        for ii in range(0, filter_height//8):
          self.kernel.write('MACC {}*{}+{}+{}, {}+{}\n'.format(h_ind, input_height, w_rnd2, ii, h_ind*(filter_height//8), ii))
      self.kernel.write('{}{} {}, {}\n\n'.format(store_op, store_n, out_y, shift))
    self.kernel.write('{}:\n'.format(kernel_end_label))

    # sequencer code
    # TODO: maybe change to take in the tensors
    self.code.write(self.layer_code(layer_label=layer_label,
                                    calc_label=calc_z_label,
                                    kernel=kernel_label, kernel_end=kernel_end_label,
                                    coef0_label=coef0_label, coef0_len=coef0_len,
                                    coef1_label=coef1_label, coef1_len=coef1_len,
                                    bias_label=bias_label,
                                    input_label=input_label,
                                    output_label=output_label))
    self.AllocateBuffer(output_tensor)

    # compute 2ker column = kernel
    # compute 2ker conv = compute column, increment VBP, SBP
    # compute depth = compute conv, increment LBP, CBP, Set VBP, increment SBP
    self.code.write('{}:\n'.format(calc_w_label))
    exec_its = output_width
    VBPinc = input_height
    if output_fmt == self.PADDED:
      SBPinc = output_height_pad
    elif output_fmt == self.PACKED:
      SBPinc = output_height
    for out_x in range(exec_its):
      self.code.write('Execute {}, {}/4 - {}/4\n'.format(0, kernel_end_label, kernel_label))
      self.code.write('AddVBP {}\n'.format(VBPinc))
      self.code.write('AddSBP {}\n'.format(SBPinc))
    self.code.write('Return\n\n')

    # output is stored kernel interleaved, This means it can't easily be used by
    # the next computtion which is likely expecting height interleaved.
    # - use RELU0 and RELU1 to store results in arbitrary location
    # - option to make coeff1 always shifted by stride_h of the kernels, biases
    #   would also need to be duplicated
    dec_len = -output_width * input_height
    inc_len = (input_width - output_width) * input_height
    self.code.write('{}:\n'.format(calc_z_label))
    for ii in range(depth_mult*input_depth):
      self.code.write('Call {}\n'.format(calc_w_label))
      # reset to start of depth, increment
      if ii%depth_mult == depth_mult-1:
        self.code.write('AddVBP {}\n'.format(inc_len))
      else:
        self.code.write('AddVBP {}\n'.format(dec_len))
      # move to next kernel and next set of biases
      self.code.write('AddCBP {}\n'.format(filter_width))
      self.code.write('AddLBP {}\n'.format(8))

    self.code.write('Return\n\n')

    return layer_label, self.PADDED

  def convert(self):

    input_tensor = self.subgraph.Tensors(self.subgraph.Inputs(0))
    if input_tensor.Buffer() and self.model.Buffers(input_tensor.Buffer()).DataLength() != 0:
      input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor)
      self.GenDataBuffer(input_tensor.Name().decode('ascii'),  input_buf)
    else:
      self.AllocateBuffer(input_tensor)

    layers = []
    output_fmt = self.PACKED
    for ii in self.traceOps():

      # TODO: abstract this to the super class or helper functions
      op = self.subgraph.Operators(ii)
      ind = op.OpcodeIndex()
      builtincode = self.model.OperatorCodes(ind).BuiltinCode()
      customcode = self.model.OperatorCodes(ind).CustomCode()
      assert (customcode is None), 'Custom operators not supported: opcode {}'.format(customcode)

      # get operation name
      op_name = [k for k,v in BuiltinOperator.__dict__.items() if v==builtincode]
      assert (len(op_name) == 1)
      op_name = op_name[0]

      # map to generation function
      op_gen = getattr(self, op_name + '_gen', None)
      assert (op_gen is not None and callable(op_gen)), 'No function to generate for op {}'.format(op_name)

      # get options structure
      opt_names = [k for k,v in BuiltinOptions.__dict__.items() if v==op.BuiltinOptionsType()]
      assert (len(opt_names) == 1)
      specopt = getattr(importlib.import_module('tflite.{}'.format(opt_names[0])), opt_names[0])()
      builtin_opt = op.BuiltinOptions()
      specopt.Init(builtin_opt.Bytes, builtin_opt.Pos)

      layer_name, output_fmt = op_gen(op, specopt, ii, output_fmt)
      if layer_name:
        layers.insert(0, layer_name)

    return self.GenOutput(layers)

def usage():
  print('{} - convert tflite model to mlaccel asm'.format(basename(sys.argv[0])))
  sys.exit(1)

if __name__ == '__main__':
  opts, rem = getopt(sys.argv[1:], 'hi:o:')
  inputf = None
  outputf = None

  for opt, arg in opts:
    if opt == '-i':
      inputf = arg
    elif opt == '-o':
      outputf = arg
    else:
      usage()

  assert (inputf), 'Input file required'
  assert (outputf), 'Output file required'

  converter = TfLiteAsmConverter(inputf)
  asm = converter.convert()

  with open(outputf, 'w') as w:
    w.write(asm.getvalue())
