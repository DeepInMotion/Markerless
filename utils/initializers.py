import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.initializers import Initializer

class BilinearWeights(Initializer):
    """Initializer that generates tensors initialized according to bilinear interpolation
    """
    
    def __init__(self, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype

    def __call__(self, shape=None, dtype=None):
        
        # Initialize parameters
        if shape:
            self.shape = shape
        self.dtype = np.float32
        filter_size = self.shape[0]
        num_channels = self.shape[2]

        # Create bilinear weights
        bilinear_kernel = np.zeros([filter_size, filter_size], dtype=self.dtype)
        scale_factor = (filter_size + 1) // 2
        if filter_size % 2 == 1:
            center = scale_factor - 1
        else:
            center = scale_factor - 0.5
        for x in range(filter_size):
            for y in range(filter_size):
                bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                        (1 - abs(y - center) / scale_factor)
        
        # Assign weights
        weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
        for i in range(num_channels):
            weights[:, :, i, i] = bilinear_kernel
        
        return K.constant(value=weights)
    
    def get_config(self):
        return {'shape': self.shape}