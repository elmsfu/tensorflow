"""
Make sure we only have conv, depthwiseconv, FC, AVGpool and maxpool layers.

Quantize everything to INT8 operations.

Strip reshape and softmax layers of the beggining and end of layer stack
"""

"""
Development plan:
# float -> uint32
# float/uint32 -> uint8 inputs,outputs, and coefficients. uint32 bias


Algorithm:
# load graph
# start at output and traverse up to input
# convert each ops, inputs (extra credit: estimate noise from quantization)
# output new buffers/tensors

"""
import flatbuffers
from getopt import getopt
import importlib
from io import StringIO
import json
import numpy as np
import re
import struct
import sys

import tflite2mlasm
import tflite_utils

import tflite
from tflite import *


class TfLiteWriterBin:
  def __init__(self):
    self.builder = flatbuffers.Builder(0)

  def build_ops(self, model):
    olen = model.OperatorCodesLength()
    ops = [model.OperatorCodes(ii) for ii in range(olen)]

    ops_vec = []
    for op in ops:
      OperatorCode.OperatorCodeStart(self.builder)
      OperatorCode.OperatorCodeAddBuiltinCode(self.builder, op.BuiltinCode())
      ops_vec.append(OperatorCode.OperatorCodeEnd(self.builder))

    Model.ModelStartOperatorCodesVector(self.builder, olen)
    for op in reversed(ops_vec):
      self.builder.PrependUOffsetTRelative(op)

    return self.builder.EndVector(olen)

  def write(self, fout, model):

    desc = self.builder.CreateString(model.Description())

    #   op_codes
    ops = self.build_ops(model)
    """
    #   subgraph
    olen = model.SubGraphsLength()
    Model.ModelStartSubGraphsVector(self.builder, olen)
    for ii in reversed(range(olen)):
      self.builder.PrependUOffsetTRelative(model.SubGraphs(ii))
    subgraphs = self.builder.EndVector(olen)

    #   buffers
    olen = model.BuffersLength()
    Model.ModelStartBuffersVector(self.builder, olen)
    for ii in reversed(range(olen)):
      self.builder.PrependUOffsetTRelative(model.Buffers(ii))
    buffers = self.builder.EndVector(olen)

    #   meta data
    olen = model.MetadataBufferLength()
    Model.ModelStartMetadataBufferVector(self.builder, olen)
    for ii in reversed(range(olen)):
      self.builder.PrependUOffsetTRelative(model.MetadataBuffer(ii))
    meta = self.builder.EndVector(olen)
    """

    # model
    Model.ModelStart(self.builder)
    Model.ModelAddVersion(self.builder, model.Version())
    Model.ModelAddDescription(self.builder, desc)
    # Model.ModelAddMetadataBuffer(self.builder, meta)
    # Model.ModelAddBuffers(self.builder, buffers)
    # Model.ModelAddSubGraphs(self.builder, subgraphs)
    Model.ModelAddOperatorCodes(self.builder, ops)

    fmodel = Model.ModelEnd(self.builder)
    self.builder.Finish(fmodel)
    # fout.write(b'\x08\x00\x00\x00')
    # fout.write(b'TFL3')
    fout.write(self.builder.Output())


class TfLiteQuant(tflite2mlasm.TFLiteConverter):
  __slots__ = []
  def init(self, fname):
    super().init(fname)


  def FULLY_CONNECTED_gen(self, op, opts, opind, output_fmt):
    return 'fc', None

  def MAX_POOL_2D_gen(self, op, opts, opind, output_fmt):
    return 'mp', None

  def CONV_2D_gen(self, op, opts, opind, output_fmt):
    return 'conv', None

  def DEPTHWISE_CONV_2D_gen(self, op, opts, opind, output_fmt):
    # dimensions
    assert (op.InputsLength() == 3)
    assert (op.OutputsLength() == 1)

    input_tensor = self.subgraph.Tensors(op.Inputs(0))
    filter_tensor = self.subgraph.Tensors(op.Inputs(1))
    bias_tensor = self.subgraph.Tensors(op.Inputs(2))
    output_tensor = self.subgraph.Tensors(op.Outputs(0))

    input_buf = tflite_utils.ConvertBuffer(self.model, input_tensor, False, False)
    filter_buf = tflite_utils.ConvertBuffer(self.model, filter_tensor, False, False)
    bias_buf = tflite_utils.ConvertBuffer(self.model, bias_tensor, False, False)

    filter_min, filter_max = np.min(filter_buf.flatten()), np.max(filter_buf.flatten())
    bias_min, bias_max = np.min(bias_buf.flatten()), np.max(bias_buf.flatten())

    # TODO: for now assume float32 and in (-1,1)
    assert (filter_min > -1 or filter_max < 1)
    assert (bias_min > -1 or bias_max < 1)

    tt = np.int8(np.round(filter_buf*127))

    return 'dconv', None

  def SOFTMAX_gen(self, op, opts, opind, output_fmt):
    return 'sm', None

  def GenOutput(self, struct):
    buf_asm = StringIO()
    # Have to sort because flatc will only scan one key past unions for the type
    buf_asm.write(json.dumps(struct, indent=2, sort_keys=True))
    return buf_asm


  def ops(self):
    def op_lut(op):
      val = [name for name, value in vars(BuiltinOperator.BuiltinOperator).items() if name.isupper() and value == op]
      return val[0]

    ret = []
    for ii in range(self.model.OperatorCodesLength()):
      op = self.model.OperatorCodes(ii)
      ret.append({'builtin_code': op_lut(op.BuiltinCode())})

    return ret

  def buffers(self):
    ret = []
    for ii in range(self.model.BuffersLength()):
      buf = self.model.Buffers(ii)
      if isinstance(buf, tflite.Buffer.Buffer):
        byte_vals = [buf.Data(jj) for jj in range(buf.DataLength())]
        ret.append({'data': byte_vals})
    return ret

  def dec_subgraph(self):
    ret = []

    def tensors(subgraph):
      ret = []
      for ii in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(ii)
        t_dict = {}
        t_dict['shape'] = tensor.ShapeAsNumpy().tolist()
        t_dict['type'] = tensor.Type()
        t_dict['buffer'] = tensor.Buffer()
        t_dict['name'] = tensor.Name().decode('ascii')
        # t_dict['quantization'] =
        ret.append(t_dict)
      return ret

    def ops(subgraph):

      def get_opt(opt_type):
        val = [name for name, value in vars(BuiltinOptions.BuiltinOptions).items() if name[0].isupper() and value == opt_type]
        return val[0]

      def get_opts(builtin_opt, opt_name):
        opt_cls = getattr(importlib.import_module('tflite.{}'.format(opt_name)), opt_name)
        specopt = opt_cls()
        specopt.Init(builtin_opt.Bytes, builtin_opt.Pos)
        res = {}
        for name, value in vars(opt_cls).items():
          if name[0].isupper() and callable(value) and not name.startswith('Init') and not name.startswith('GetRoot'):
            t1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            dname = re.sub('([a-z0-9])([A-Z])', r'\1_\2', t1).lower()
            res[dname] = getattr(specopt, name)()
        return res

      ret = []
      for ii in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(ii)
        op_dict = {}
        op_dict['inputs'] = op.InputsAsNumpy().flatten().tolist()
        op_dict['outputs'] = op.OutputsAsNumpy().flatten().tolist()
        op_dict['opcode_index'] = op.OpcodeIndex()

        op_dict['builtin_options_type'] = get_opt(op.BuiltinOptionsType())
        op_dict['builtin_options'] = get_opts(op.BuiltinOptions(), op_dict['builtin_options_type'])
        op_dict['mutating_variable_inputs'] = []
        ret.append(op_dict)
      return ret

    for ii in range(self.model.SubgraphsLength()):
      sg = self.model.Subgraphs(ii)
      sg_dict = {}
      sg_dict['tensors'] = tensors(sg)
      sg_dict['inputs'] = sg.InputsAsNumpy().flatten().tolist()
      sg_dict['outputs'] = sg.OutputsAsNumpy().flatten().tolist()
      sg_dict['operators'] = ops(sg)
      name = sg.Name()
      if name:
        sg_dict['name'] = name.decode('ascii')
      ret.append(sg_dict)
    return ret

  def getStruct(self):
    top = {}
    top['version'] = self.model.Version()
    top['operator_codes'] = self.ops()
    top['subgraphs'] = self.dec_subgraph()
    top['buffers'] = self.buffers()
    top['description'] = self.model.Description().decode('ascii')

    return top

  def convert(self):
    top = self.getStruct()

    sg = top['subgraphs'][0]
    for ii in self.traceOps():
      op = sg['operators'][ii]
      if len(op['inputs']) > 2:
        u8_inputs = op['inputs'][:2]
        i32_inputs = op['inputs'][2:]
      else:
        u8_inputs = op['inputs']
        i32_inputs = []

      for inp in u8_inputs:
        buf_ind = sg['tensors'][inp]['buffer']
        sg['tensors'][inp]['type'] = TensorType.TensorType.UINT8

        tensor = self.subgraph.Tensors(inp)
        buf = tflite_utils.ConvertBuffer(self.model, tensor, False, False)
        if buf is not None:
          tt = np.uint8(np.round(buf*127) + 127)

          # TODO: for now assume float32 and in (-1,1)
          #assert (data_min > -1 or data_max < 1)
          tflat = tt.flatten()
          top['buffers'][buf_ind]['data'] = struct.unpack('{}B'.format(1*len(tflat)), tflat.tobytes())

      for inp in i32_inputs:
        buf_ind = sg['tensors'][inp]['buffer']
        sg['tensors'][inp]['type'] = TensorType.TensorType.INT32

        tensor = self.subgraph.Tensors(inp)
        buf = tflite_utils.ConvertBuffer(self.model, tensor, False, False)
        if buf is not None:
          tt = np.int32(np.round(buf*127))

          # TODO: for now assume float32 and in (-1,1)
          #assert (data_min > -1 or data_max < 1)
          tflat = tt.flatten()
          top['buffers'][buf_ind]['data'] = struct.unpack('{}B'.format(4*len(tflat)), tflat.tobytes())

    return self.GenOutput(top)

def main():
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

  converter = TfLiteQuant(inputf)
  res = converter.convert()

  with open(outputf, 'w') as w:
    w.write(res.getvalue())


if __name__ == '__main__':
  main()
