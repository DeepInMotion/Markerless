from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Lambda, Conv2D, BatchNormalization, Activation, Multiply, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils.activations import Swish, eswish
    
get_custom_objects().update({
    'Swish': Swish,
})

def se_eswish(inputs, se_ratio=24, prefix=''):
    if K.image_data_format() == "channels_first":
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]
    input_filters = inputs.get_shape()[channel_axis]#_keras_shape[channel_axis]
    num_reduced_filters = max(1, int(input_filters / se_ratio))
        
    # Squeeze-and-Excitation  
    
    # Squeeze
    x = inputs
    x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True), name=prefix+'se_squeeze_lambda')(x)
    x = Conv2D(num_reduced_filters, 1, padding='same', use_bias=True, name=prefix+'se_squeeze_conv')(x)
    x = Swish('eswish', name=prefix+'se_squeeze_eswish')(x)
    
    # Excite
    x = Conv2D(input_filters, 1, padding='same', use_bias=True, name=prefix+'se_excite_conv')(x)
    x = Activation('sigmoid', name=prefix+'se_excite_sigmoid')(x)
    out = Multiply(name=prefix+'se_multiply')([x, inputs])
    
    return out