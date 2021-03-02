import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import math
import numpy as np
import timeit

import utils.initializers as initializers

def upscale_block(raw_output, num_body_parts, raw_output_resolution, upscaled_output_resolution):
    scale_factor = upscaled_output_resolution / raw_output_resolution
    num_transpose = int(math.log(scale_factor, 2))
    transposeX = raw_output
    for i in range(1, num_transpose + 1): 
        if i == num_transpose:
            transposeX = Conv2DTranspose(num_body_parts, 4, strides=(2, 2), name='upscaled_confs', padding='same', kernel_initializer=initializers.BilinearWeights())(transposeX)
        else:
            transposeX = Conv2DTranspose(num_body_parts, 4, strides=(2, 2), name='upscaled_transpose{}'.format(i), padding='same', kernel_initializer=initializers.BilinearWeights())(transposeX)

    return transposeX

def get_flops(concrete_func):    
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)

        return flops.total_float_ops
    
def summary(raw_model, upscaled_output_resolution=False, flops=True, latency=True):
         
    # Construct model with upscaling
    if upscaled_output_resolution:
        raw_output = raw_model.layers[-1].output
        raw_output_resolution = raw_output.shape[1]
        num_body_parts = raw_output.shape[3]
        upscaled_output = upscale_block(raw_output, num_body_parts, raw_output_resolution, upscaled_output_resolution)
        model = keras.Model(raw_model.inputs, upscaled_output)
    else:
        model = raw_model
        
    # Compute FLOPs
    if flops:
        flop_model = keras.Model(
            inputs=model.inputs,
            outputs=[layer.output for layer in model.layers],
        )
        concrete = tf.function(lambda inputs: flop_model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

        num_flops = get_flops(concrete_func)
    else:
        num_flops = False
        
    # Compute inference latency
    if latency:
        times = []
        batch_size = 10
        input_shape = [*model.inputs[0].shape[1:]]
        input_batch = np.random.rand(batch_size, input_shape[0], input_shape[1], input_shape[2])
        for i in range(10):
            start_time = timeit.default_timer()
            output = model.predict(input_batch)
            end_time = timeit.default_timer()
            ms = (end_time - start_time)*1000
            times.append(ms)
        sorted_times = sorted(times)
        num_ms = sorted_times[4]/batch_size
        fps = 1/(num_ms/1000)
    else:
        num_ms = False
    
    # Determine devices
    gpus = [gpu.name for gpu in tf.config.list_physical_devices('GPU')]
    cpus = [cpu.name for cpu in tf.config.list_physical_devices('CPU')]
    devices = 'GPU(s): {0}, CPU(s): {1}'.format(gpus, cpus)
        
    # Display Keras model summary
    model.summary(200)
    num_parameters = model.count_params()
    
    # Display FLOPs
    if flops:
        print('Floating point operations: {0}'.format(num_flops))
        print('____________________________________________________________________________________________________________________________________________')
        
    # Display inference latency in milliseconds
    if latency:
        print('Inference latency: {0:.1f} ms'.format(num_ms))
        print('Frames per second: {0:.3f}'.format(fps))
        print('Devices: {0}'.format(devices))
        print('____________________________________________________________________________________________________________________________________________')
    
    return num_parameters, num_flops, num_ms, devices