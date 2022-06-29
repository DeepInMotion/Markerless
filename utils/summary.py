import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from scipy.ndimage.filters import gaussian_filter
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

def gauss_smooth_soft_argmax(output, num_body_parts, batch_num):
    soft_argmax = True
    batch = []
    threshold = 0.0001
    confs = np.array(output)
    bdp_factor = {0: 1.04, 1: 0.53, 2: 1.04, 3: 0.93, 4: 1.00, 5: 1.01, 6: 0.81, 7: 0.71, 8: 1.23, 9: 1.08, 10: 0.79, 11: 0.70, 12: 1.49, 13: 1.55, 14: 1.06, 15: 0.78, 16: 1.60, 17: 0.93, 18: 0.77}
    for n in range(0,batch_num):
        points = []
        for idx in range(0,num_body_parts):
            conf = confs[n, ..., idx]

            # conf: 2d confidence map of shape (height, width)
            output_width = int(conf.shape[1])
            output_height = int(conf.shape[0])
            sigma = bdp_factor[idx]*(output_width/32) 

            # Apply gaussian filter
            #gauss = gaussian_filter(conf, sigma=sigma)

            # Locate peaks in confidence map
            max_index = np.argmax(conf)
            peak_y = math.floor(max_index / output_width)
            peak_x = max_index % output_width

            # Verify confidence of prediction
            conf_value = conf[int(peak_y),int(peak_x)]
            if conf_value < threshold:
                peak_x = 0.0
                peak_y = 0.0
            else:
                if soft_argmax:
                    #Local soft-argmax
                    # Define beta and size of local square neighborhood
                    beta = 9.
                    num_pix = int(np.round(2*sigma))
                    num_pix1 = int(num_pix+1)
                    # Define start and end indx for local square neighborhood
                    rows_start = int(np.max([int(peak_y)-num_pix, 0]))
                    rows_end = int(np.min([int(peak_y)+num_pix1, output_height]))
                    cols_start = int(np.max([int(peak_x)-num_pix, 0]))
                    cols_end = int(np.min([int(peak_x)+num_pix1, output_width]))
                    # Define localsquare neigborhod 
                    loc_mat = conf[rows_start:rows_end,cols_start:cols_end]
                    y_ind = [i for i in range(rows_start,rows_end)]
                    x_ind = [j for j in range(cols_start,cols_end)]
                    posy, posx = np.meshgrid(y_ind, x_ind, indexing='ij')
                    # Compute local soft-argmax for neigborhood
                    a = np.exp(beta*loc_mat)
                    b = np.sum(a)
                    softmax = a/b
                    peak_x = np.sum(softmax*posx)
                    peak_y = np.sum(softmax*posy)
                peak_x += 0.5
                peak_y += 0.5

            peak_x /= output_width
            peak_y /= output_height
            points.append([peak_x, peak_y])
        
        batch.append(points)
    batch_out = np.asarray(batch)     
    return batch_out
                              
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
    
    model_output = model.layers[-1].output
    num_body_parts = model_output.shape[3]
        
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
            #print(np.shape(output))
            batch_out = gauss_smooth_soft_argmax(output, num_body_parts, batch_size)
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