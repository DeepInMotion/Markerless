from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, BatchNormalization, Concatenate, DepthwiseConv2D, Add, Conv2DTranspose
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import units, initializers
from utils.activations import Swish, eswish

from models.efficientnets.efficientnetv1.tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from models.efficientnets.efficientnetv1.tfkeras import preprocess_input as efficientnet_preprocess_input

class architecture:
    def __init__(self, input_resolution, num_body_parts, num_segments, output_resolution=None):
        """ Initializes EfficientPose model. """
        
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
        self.num_resolutions = {128: 1, 224: 1, 256: 2, 368: 2, 480: 2, 600: 2}[self.input_resolution]
        self.output_types = [True, None, None]
        self.num_skeleton_passes = 1
        self.num_separate_passes = 2
        self.output_names = {True: 'pafs', False: 'segments', None: 'confs'}
        self.num_body_parts = num_body_parts
        self.num_segments = num_segments
        self.output_channels = {True: 2*self.num_segments, False: self.num_segments, None: self.num_body_parts}
        self.efficientnet_features = {128: {128: (69, 41)}, 224: {224: (69, 41)}, 256: {256: (111, 68), 128: (42, 14)}, 368: {368: (141, 83), 184: (42, 14)}, 480: {480: (183, 110), 240: (69, 26)}, 600: {600: (255, 152), 300: (69, 26)}}[self.input_resolution]
        self.detection_channels = {128: 40, 224: 40, 256: 48, 368: 56, 480: 64, 600: 80}[self.input_resolution]
        self.detection_blocks = {128: 1, 224: 1, 256: 1, 368: 2, 480: 3, 600: 4}[self.input_resolution]
        self.expansion_rate = {128: 6, 224: 6, 256: 6, 368: 6, 480: 6, 600: 6}[self.input_resolution]
        self.pretrained_path = {128: os.path.join('models/pretrained', 'MPII_128x128_EfficientPoseDet_weights.hdf5'), 
                                224: os.path.join('models/pretrained', 'MPII_224x224_EfficientPoseRT_weights.hdf5'), 
                                256: os.path.join('models/pretrained', 'MPII_256x256_EfficientPoseI_weights.hdf5'), 
                                368: os.path.join('models/pretrained', 'MPII_368x368_EfficientPoseII_weights.hdf5'),
                                480: os.path.join('models/pretrained', 'MPII_480x480_EfficientPoseIII_weights.hdf5'), 
                                600: os.path.join('models/pretrained', 'MPII_600x600_EfficientPoseIV_weights.hdf5')}[self.input_resolution]
        
        # Define inputs of multiple resolutions
        
        # Initialize inputs
        self.inputs = {}

        # Full-resolution image
        input_shape = (self.input_resolution, self.input_resolution, 3)
        input_layer = Input(shape=input_shape, name='input_res1')
        self.inputs['res1'] = input_layer

        # Downscaled images
        for n in range(2, self.num_resolutions + 1):
            input_layer = AveragePooling2D(pool_size=(2, 2), padding='same', name='input_res{0}'.format(n))(input_layer)
            self.inputs['res{0}'.format(n)] = input_layer
            
        # Feature extractors    
        
        # Initialize individual feature extractors
        feature_extractors = {}

        # Create separate feature extractor per resolution
        for n in range(1, self.num_resolutions + 1):

            # Relevant input
            input_layer = self.inputs['res{0}'.format(n)]

            # Pre-trained EfficientNetB5
            input_height = self.input_resolution / n
            if input_height < 232:
                efficientnet_module = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None) 
            elif 232 <= input_height < 250:
                efficientnet_module = EfficientNetB1(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)  
            elif 250 <= input_height < 280:
                efficientnet_module = EfficientNetB2(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)   
            elif 280 <= input_height < 340:
                efficientnet_module = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif 340 <= input_height < 418:
                efficientnet_module = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif 418 <= input_height < 492:
                efficientnet_module = EfficientNetB5(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif 492 <= input_height < 564:
                efficientnet_module = EfficientNetB6(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif 564 <= input_height:
                efficientnet_module = EfficientNetB7(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)

            for layer in efficientnet_module.layers: 
                if not layer.name.startswith('input'):
                    layer._name += '_res{0}'.format(n)
            for i, layer_index in enumerate(self.efficientnet_features[input_height], 1):
                feature_extractor = efficientnet_module.layers[layer_index].output
                feature_extractors['res{0}_{1}'.format(n, i)] = feature_extractor
        
        # Construct combined feature extractor for detection
        
        if self.num_resolutions > 1:

            # Fetch feature extractors
            all_feature_extractors = []
            for res in feature_extractors.keys():
                if res.endswith('_1'):
                    all_feature_extractors.append(feature_extractors[res])
            
            # Concatenate feature extractors
            detection_feature_extractors = all_feature_extractors

        else:
            detection_feature_extractors = [feature_extractors['res1_1']]
                 
        # Detection stage

        # Initialize model outputs
        self.model_outputs = []

        # Skeleton branch
        if self.num_resolutions > 1:
            feature_layer = Concatenate()(detection_feature_extractors)
        else:
            feature_layer = detection_feature_extractors[0]
    
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
            passT_concat = Concatenate()([passT_features] + detection_feature_extractors)
            passT_pred, passT_features = self.passT_block(passT_concat, blocks=self.detection_blocks, output_type=self.output_types[i-1], pass_number=i, expansion_rate=self.expansion_rate, branch_type='detection{0}'.format(i - self.num_skeleton_passes))
            self.model_outputs.append(passT_pred)
        
        # Upscale stage
        if self.upscale:
            deconvX = self.upscale_block(detection_layer=passT_pred)
            self.model_outputs = deconvX
            
        # Build EfficientPose model
        self.model = Model(self.inputs['res1'], self.model_outputs, name="EfficientPose")

        # Freeze detector layers
        if self.upscale:
            for layer in self.model.layers: layer.trainable = False
            
    def mbconv_block(self, inputs, channels, kernel_size=5, dilation_rate=1, expansion_rate=6, dense=True, prefix=''):
        
        # Expand features
        name = prefix + '_conv1'
        conv1 = Conv2D(channels*expansion_rate, 1, name=name, padding='same', use_bias=False)(inputs)
        conv1 = BatchNormalization(name=name+'_bn')(conv1)
        conv1 = Swish('eswish', name=name+'_eswish')(conv1)
                    
        # Depthwise convolution
        name = prefix + '_dconv1'
        dconv = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, depth_multiplier=1, name=name, padding='same', use_bias=False)(conv1)
        dconv = BatchNormalization(name=name+'_bn')(dconv)
        dconv = Swish('eswish', name=name+'_eswish')(dconv)
            
        # Squeeze-and-Excitation
        name = prefix + '_se'
        se = units.se_eswish(dconv, prefix=name+'_')

        # Bottleneck
        name = prefix + '_conv2'
        conv2 = Conv2D(channels, 1, name=name, padding='same', use_bias=False)(se)
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
        Numpy array pre-processed according to EfficientNet standard
    """
    
    return efficientnet_preprocess_input(x)

if __name__ == '__main__':
    efficientpose = architecture(input_resolution=224, num_body_parts=19, num_segments=36)
    efficientpose.model.summary(140)