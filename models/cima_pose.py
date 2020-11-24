from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.applications import densenet
from tensorflow.keras import backend as K
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import initializers 

class architecture:

    def __init__(self, input_resolution, num_body_parts, num_segments=None, output_resolution=None):
        """ Initializes CIMA-Pose model. """
        
        # Initialize parameters
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.scale_factor = 8 
        self.upscale = True
        if output_resolution == None:
            self.upscale = False
            self.output_size = (int(self.input_resolution/self.scale_factor), int(self.input_resolution/self.scale_factor))
        else:
            self.output_size = (self.output_resolution, self.output_resolution)
        self.num_passes = 2
        self.num_body_parts = num_body_parts
        self.num_segments = num_segments
        self.densenet_features = 'pool3_conv'
        self.pretrained_path = {368: os.path.join('models/pretrained', 'MPII_368x368_CIMA-Pose_weights.hdf5')}[self.input_resolution]
         
        # Normalized input layer
        input_shape = (self.input_resolution, self.input_resolution, 3)
        input_raw = Input(shape=input_shape)
        
        # Pre-trained DenseNet-121: First 2 dense blocks
        densenet_module = densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=input_raw, input_shape=None, pooling=None)   
        pool3_relu = densenet_module.get_layer(self.densenet_features).output
            
        # Final step of feature extraction
        conv4 = Conv2D(128, 3, name='features_MPII', activation='relu', padding='same')(pool3_relu)

        # Detection head
        stageX_conv6 = None
        if not self.upscale:
            model_outputs = []
            for stage in range(1, self.num_passes + 1):
                stageX_conv6 = self.stageT_block(feature_layer=conv4, prev_output=stageX_conv6, stage=stage)
                model_outputs.append(stageX_conv6)
        else:
            for stage in range(1, self.num_passes + 1):
                stageX_conv6 = self.stageT_block(feature_layer=conv4, prev_output=stageX_conv6, stage=stage)
            deconvX = self.upscale_block(detection_layer=stageX_conv6)
            model_outputs = deconvX
            
        # Build CIMA-Pose model
        self.model = Model(input_raw, model_outputs, name='cima-pose')
        
        # Freeze detector layers
        if self.upscale:
            for layer in self.model.layers: layer.trainable = False

    def stageT_block(self, feature_layer, prev_output, stage):
        
        # Match receptive field of the previous output layer 
        if stage > 1:
            concat_layer = concatenate([prev_output, feature_layer])
        else:
            concat_layer = feature_layer
            
        # Perform detection
        stageX_conv1 = Conv2D(128, 3, name='stage{}_conv1_tune'.format(stage), activation='relu', padding='same')(concat_layer)
        stageX_conv2 = Conv2D(64, 3, name='stage{}_conv2'.format(stage), activation='relu', padding='same')(stageX_conv1)
        stageX_conv3_dilated = Conv2D(64, 3, name='stage{}_conv3_dilated'.format(stage), activation='relu', padding='same', dilation_rate=2)(stageX_conv2)
        stageX_conv4_dilated = Conv2D(64, 3, name='stage{}_conv4_dilated'.format(stage), activation='relu', padding='same', dilation_rate=4)(stageX_conv3_dilated)
        stageX_conv5 = Conv2D(64, 1, name='stage{}_conv5'.format(stage), activation='relu', padding='same')(stageX_conv4_dilated)
        stageX_conv6 = Conv2D(self.num_body_parts, 1, name='stage{}_confs_tune'.format(stage), padding='same')(stageX_conv5)

        return stageX_conv6
    
    def upscale_block(self, detection_layer):
        
        # Perform upscaling
        scale_factor = (self.output_size[0] / self.input_resolution) * self.scale_factor
        num_deconvs = int(math.log(scale_factor, 2))
        deconvX = detection_layer
        for i in range(1, num_deconvs + 1): 
            if i == num_deconvs:
                deconvX = Conv2DTranspose(self.num_body_parts, 4, strides=(2, 2), name='upscaled_confs', padding='same', kernel_initializer=initializers.BilinearWeights())(deconvX)
            else:
                deconvX = Conv2DTranspose(self.num_body_parts, 4, strides=(2, 2), name='upscaled_deconv{}'.format(i), padding='same', kernel_initializer=initializers.BilinearWeights())(deconvX)

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
        Numpy array pre-processed according to DenseNet standard
    """
    
    return densenet.preprocess_input(x)

if __name__ == '__main__':
    cima_pose = architecture(input_resolution=368, num_body_parts=19)
    cima_pose.model.summary()