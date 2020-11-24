from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, DepthwiseConv2D, Add, Conv2DTranspose
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import initializers

from models.efficientnet_lite import efficientnet_lite

class architecture:
    def __init__(self, input_resolution, num_body_parts, num_segments, output_resolution=None):
        """ Initializes EfficientPose Lite model. """
        
        # Variables
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.scale_factor = 8 
        self.upscale = True
        if output_resolution == None:
            self.upscale = False
            self.output_size = (int(self.input_resolution/self.scale_factor), int(self.input_resolution/self.scale_factor))
        else:
            self.output_size = (self.output_resolution, self.output_resolution)
        self.output_types = [True, None, None]
        self.num_skeleton_passes = 1
        self.num_separate_passes = 2
        self.output_names = {True: 'pafs', False: 'segments', None: 'confs'}
        self.num_body_parts = num_body_parts
        self.num_segments = num_segments
        self.output_channels = {True: 2*self.num_segments, False: self.num_segments, None: self.num_body_parts}
        self.efficientnet_lite_variant = {128: 0, 224: 0, 256: 2, 368: 4}[self.input_resolution] #L0, L0, L2, L4
        self.efficientnet_lite_features = {128: {128: (42, 25)}, 224: {224: (42, 25)}, 256: {256: (60, 34)}, 368: {368: (78, 43)}}[self.input_resolution]
        self.detection_channels = {128: 40, 224: 40, 256: 48, 368: 56}[self.input_resolution]
        self.detection_blocks = {128: 1, 224: 1, 256: 1, 368: 2}[self.input_resolution]
        self.expansion_rate = {128: 6, 224: 6, 256: 6, 368: 6}[self.input_resolution]
        self.pretrained_path = {128: os.path.join('models/pretrained', 'MPII_128x128_EfficientPoseDetLite_weights.hdf5'), 
                                224: os.path.join('models/pretrained', 'MPII_224x224_EfficientPoseRTLite_weights.hdf5'), 
                                256: os.path.join('models/pretrained', 'MPII_256x256_EfficientPoseILite_weights.hdf5'),
                                368: os.path.join('models/pretrained', 'MPII_368x368_EfficientPoseIILite_weights.hdf5')}[self.input_resolution]
                        
        # Feature extractor   
        efficientnet_lite_module = efficientnet_lite.efficientnet_lite(self.efficientnet_lite_variant, input_resolution=self.input_resolution).model
        input_layer = efficientnet_lite_module.layers[0].input
                 
        # Detection stage

        # Initialize model outputs
        self.model_outputs = []

        # Skeleton branch
        feature_layer = efficientnet_lite_module.layers[self.efficientnet_lite_features[self.input_resolution][0]].output
    
        passT_features = feature_layer
        for i in range(1, self.num_skeleton_passes + 1):
            if i == 1:
                passT_pred, passT_features = self.passT_block(passT_features, blocks=self.detection_blocks, output_type=self.output_types[i-1], pass_number=i, expansion_rate=self.expansion_rate, branch_type='skeleton')
            else:
                passT_concat = Concatenate()([passT_features, feature_layer])
                passT_pred, passT_features = self.passT_block(passT_concat, blocks=self.detection_blocks, output_type=self.output_types[i-1], pass_number=i, expansion_rate=self.expansion_rate, branch_type='skeleton')
            self.model_outputs.append(passT_pred)
    
        # Branch for keypoint estimation
        for i in range(self.num_skeleton_passes + 1, self.num_skeleton_passes + self.num_separate_passes + 1):
            passT_concat = Concatenate()([passT_features] + [feature_layer])
            passT_pred, passT_features = self.passT_block(passT_concat, blocks=self.detection_blocks, output_type=self.output_types[i-1], pass_number=i, expansion_rate=self.expansion_rate, branch_type='detection{0}'.format(i - self.num_skeleton_passes))
            self.model_outputs.append(passT_pred)
        
        # Upscale stage
        if self.upscale:
            deconvX = self.upscale_block(detection_layer=passT_pred)
            self.model_outputs = deconvX
        
        # Build EfficientPose model
        self.model = Model(input_layer, self.model_outputs, name="EfficientPoseLite")

        # Freeze detector layers
        if self.upscale:
            for layer in self.model.layers: layer.trainable = False
            
    def mbconv_block(self, inputs, channels, kernel_size=5, dilation_rate=1, expansion_rate=6, dense=True, prefix=''):
        
        # Expand features
        name = prefix + '_conv1'
        conv1 = Conv2D(channels*expansion_rate, 1, name=name, padding='same', use_bias=False)(inputs)
        conv1 = BatchNormalization(name=name+'_bn')(conv1)
        conv1 = ReLU(max_value=6, name=name+'_relu6')(conv1)
                    
        # Depthwise convolution
        name = prefix + '_dconv1'
        dconv = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, depth_multiplier=1, name=name, padding='same', use_bias=False)(conv1)
        dconv = BatchNormalization(name=name+'_bn')(dconv)
        dconv = ReLU(max_value=6, name=name+'_relu6')(dconv)
            
        # Bottleneck
        name = prefix + '_conv2'
        conv2 = Conv2D(channels, 1, name=name, padding='same', use_bias=False)(dconv)
        conv2 = BatchNormalization(name=name+'_bn')(conv2)

        # Dense connection
        if dense:
            conv2 = Concatenate(name=prefix+'_dense')([conv2, inputs])

        return conv2
    
    def passT_block(self, last_layer, blocks=3, output_type=None, pass_number=1, expansion_rate=6, branch_type='combined'):
        
        # Construct dense blocks
        passT_prev_mbconv3 = last_layer
        for n in range(1, blocks + 1):
                        
            # First depthwise convolution
            passT_blockX_mbconv1 = self.mbconv_block(passT_prev_mbconv3, self.detection_channels, kernel_size=5, expansion_rate=expansion_rate, dense=False, prefix='pass{0}_block{1}_mbconv1_{2}'.format(pass_number, n, branch_type))

            # Second depthwise convolution
            passT_blockX_mbconv2 = self.mbconv_block(passT_blockX_mbconv1, self.detection_channels, kernel_size=5, expansion_rate=expansion_rate, dense=True, prefix='pass{0}_block{1}_mbconv2_{2}'.format(pass_number, n, branch_type))

            # Third depthwise convolution
            passT_blockX_mbconv3 = self.mbconv_block(passT_blockX_mbconv2, self.detection_channels, kernel_size=5, expansion_rate=expansion_rate, dense=True, prefix='pass{0}_block{1}_mbconv3_{2}'.format(pass_number, n, branch_type))
            
            # Residual connection
            if n > 1:
                passT_prev_mbconv3 = Add(name='pass{0}_block{1}_skip_{2}'.format(pass_number, n, branch_type))([passT_blockX_mbconv3, passT_prev_mbconv3])
            else:
                passT_prev_mbconv3 = passT_blockX_mbconv3
            
        # Prediction
        passT_conv = Conv2D(128, 1, name='pass{0}_{1}_conv_tune'.format(pass_number, branch_type), padding='same')(passT_prev_mbconv3)
        passT_pred = Conv2D(self.output_channels[output_type], 1, name='pass{0}_{1}_{2}_tune'.format(pass_number, branch_type, self.output_names[output_type]), padding='same')(passT_conv)

        return passT_pred, passT_prev_mbconv3
    
    def upscale_block(self, detection_layer):
    
        # Perform upscaling
        scale_factor = (self.output_size[0] / self.input_resolution) * self.scale_factor
        num_deconvs = int(math.log(scale_factor, 2))
        deconvX = detection_layer
        for i in range(1, num_deconvs + 1): 
            if i == num_deconvs:
                deconvX = Conv2DTranspose(self.num_body_parts, 4, strides=(2, 2), name='upscaled_confs', padding='same', kernel_initializer=initializers.BilinearWeights())(deconvX)
            else:
                deconvX = Conv2DTranspose(self.num_body_parts, 4, strides=(2, 2), name='upscaled_transpose{}'.format(i), padding='same', kernel_initializer=initializers.BilinearWeights())(deconvX)

        return deconvX
    
def preprocess_input(x):
    """ Preprocesses a Numpy array encoding a batch of images. 
    
    Parameters
    ----------
    x: ndarray
        Numpy array of shape (n, h, w, 3) with RGB format
    
    Returns
    ----------
    preprocessed: ndarray
        Numpy array pre-processed according to EfficientNet Lite standard
    """
    
    return efficientnet_lite.preprocess_input(x)

if __name__ == '__main__':
    efficientposelite = architecture(input_resolution=224, num_body_parts=19, num_segments=36)
    efficientposelite.model.summary(140)