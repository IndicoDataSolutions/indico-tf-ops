import os
import warnings
import math

import tensorflow as tf
from tensorflow.python.framework import ops

from finetune.util.shapes import shape_list

try:
    kernels_module = tf.load_op_library(
        os.path.join(os.path.dirname(__file__),'../build/libindico_kernels.so')
    )
    BUILT = True
except:
    if tf.test.is_built_with_cuda():
        warnings.warn("Cuda appears to be available but cannot load the kernels")
    BUILT = False

@ops.RegisterGradient("RecursiveAgg")
def _grad_ra(op, grad0, grad1):
    return kernels_module.recursive_agg_grad(grad0, op.inputs[0], op.outputs[1])

def recursive_agg_op(inp, kernel_length, pool_len):
    return kernels_module.recursive_agg(
        inp, kernel_length, pool_len, int(math.ceil(math.log(pool_len, kernel_length)))
    )[0]

@ops.RegisterGradient("DynamicConvolution")
def _grad_dc(op, grad_output):
    return kernels_module.dynamic_convolution_grad(
        grad_output, op.inputs[0], op.inputs[1], op.get_attr("padding_l")
    )

def dynamic_convolution_op(inp, weights, padding="causal"):
    """
    inp : Batch Time Channels
    weights: Batch, Time, FilterWidth, Heads
    padding: causal or same
    """
    batch, seq, n_channels = shape_list(inp)
    _, _, kernel_width, n_heads = shape_list(weights)
    assert n_channels % n_heads == 0
    
    if padding.lower() == "causal":
        padding_l = kernel_width - 1
    elif padding.lower() == "same":
        padding_l = kernel_width // 2
    inp_formatted = tf.transpose(inp, [0, 2, 1])
    weights_formatted = tf.transpose(weights, [0, 3, 2, 1])
    return tf.transpose(kernels_module.dynamic_convolution(inp_formatted, weights_formatted, padding_l), [0, 2, 1])
    
    
    
                               
