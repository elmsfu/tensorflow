from tflite.TensorType import TensorType
import numpy as np


TF2NP = {
    TensorType.BOOL: np.bool,
    TensorType.COMPLEX64: np.complex64,
    TensorType.FLOAT16: np.float16,
    TensorType.FLOAT32: np.float32,
    TensorType.INT16: np.int16,
    TensorType.INT32: np.int32,
    TensorType.INT64: np.int64,
    TensorType.STRING: str,
    TensorType.UINT8: np.uint8
}

def ConvertBuffer(model, tensor, bFill=True, bTranspose=True, bSqueeze=True, transpose=None):
  tshape = tensor.ShapeAsNumpy()
  quant_params = tensor.Quantization()
  assert (quant_params.ZeroPointLength() in [0,1])

  if quant_params.ZeroPointLength() == 1:
    zero_point = quant_params.ZeroPoint(0)
  else:
    zero_point = 0

  tbuf = model.Buffers(tensor.Buffer()).DataAsNumpy() - zero_point
  if type(tbuf) in [np.ndarray, np.array]:
    assert(np.prod(tshape) * np.dtype(TF2NP[tensor.Type()]).itemsize == len(tbuf))
    tbuf2 = np.ndarray(tshape, buffer=tbuf.tobytes(), dtype=TF2NP[tensor.Type()])
    res = tbuf2.copy()
    # TODO: consider optimizing if a dimension is 1, same with convolution filters
    # transpose changes the default NHWC -> CWHN, we then squeeze the N away so we have CWH
    if bTranspose or transpose is not None:
      res = res.transpose(transpose)
      if bSqueeze and len(res.shape) > 1 and res.shape[-1] == 1:
        res = res.squeeze(-1)
    if bFill:
      nzfill = 8 - res.shape[-1]%8
      if nzfill != 8:
        res = np.concatenate( (res, np.zeros(res.shape[:-1] + (nzfill,), dtype=res.dtype)),
                              len(res.shape)-1)

    print('Converted Buffer {}: {}'.format(tensor.Name().decode('ascii'), res.shape))
    return res
