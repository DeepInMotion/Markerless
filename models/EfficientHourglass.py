import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ReLU, Conv2D, DepthwiseConv2D, Conv2DTranspose, Concatenate, BatchNormalization, Add, Multiply, GlobalAveragePooling2D, Activation, Reshape, UpSampling2D
import math

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import losses, initializers 

from utils.activations import Swish
from tensorflow.python.keras import backend as K

from models.efficientnet_lite import efficientnet_lite
#from models.efficientnet_X import efficientnet_X 
from models.efficientnets.efficientnetv1.keras import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4 
from models.efficientnets.efficientnetv1.keras import preprocess_input as efficientnet_preprocess_input

class architecture:
    def __init__(self, input_resolution, num_body_parts, num_segments, output_resolution=None, architecture_type='B', efficientnet_variant=0, block_variant='Block1to6', GPU=None):
        """ Initializes CIMA-Pose model. """
        # Architecture parameters
        self.model_type = architecture_type
        self.efficientnet_variant = efficientnet_variant
        self.block_variant = block_variant
        self.GPU = GPU
        
       # Variables
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_body_parts = num_body_parts
        self.scale_factor = 4 
        self.upscale = True
        if output_resolution == None:
            self.upscale = False
            self.output_size = (int(self.input_resolution/self.scale_factor), int(self.input_resolution/self.scale_factor))
        else:
            self.output_size = (self.output_resolution, self.output_resolution)
        self.pretrained_path = os.path.join('models/pretrained', 'MPII_{}x{}_EfficientHourglass{}{}_{}_weights{}.hdf5'.format('' + str(self.input_resolution), str(self.input_resolution), self.model_type, str(self.efficientnet_variant), self.block_variant, self.GPU))
        
        # Initialize parameters and weights
        trainable = True
        Bridge_block = []
        map_numb_block = []
        MBconv_factor = [6, 6, 6]
        SE_ratio = [3, 6, 12]
        
        CONV_KERNEL_INITIALIZER1 = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 1.0,
                'mode': 'fan_in',
                'distribution': 'normal'
            }
        }
        
         # Initialize model outputs
        self.model_outputs = []
        
        # Initilize EfficientNet back-end and input layer
        if(self.model_type = 'L' or 'H'):
            efficientnet_module = efficientnet_lite.efficientnet_lite(self.efficientnet_variant).model 
            input_layer = efficientnet_module.layers[0].input
            self.efficientnet_features = {0: {'res1': 25, 'res2': 42, 'res3': 94, 'res4': 129}, 1: {'res1': 34, 'res2': 60, 'res3': 130, 'res4': 174}, 2: {'res1': 34, 'res2': 60, 'res3': 130, 'res4': 174}, 3: {'res1': 34, 'res2': 60, 'res3': 148, 'res4': 201}, 4: {'res1': 43, 'res2': 78, 'res3': 184, 'res4': 255}}.[self.efficientnet_variant] 
        elif(self.model_type = 'X'): #TO BE GENERATED
            efficientnet_module = efficientnet_X.efficientnet_lite(self.efficientnet_variant).model #TO BE GENERATED 
            input_layer = efficientnet_module.layers[0].input
            #self.efficientnet_features = {0: {'res1': 25, 'res2': 42, 'res3': 94, 'res4': 129}, 1: {'res1': 34, 'res2': 60, 'res3': 130, 'res4': 174}, 2: {'res1': 34, 'res2': 60, 'res3': 130, 'res4': 174}, 3: {'res1': 34, 'res2': 60, 'res3': 148, 'res4': 201}, 4: {'res1': 43, 'res2': 78, 'res3': 184, 'res4': 255}}.[self.efficientnet_variant]
        elif(self.model_type = 'B')
            input_shape = (None, None, 3)
            input_layer = Input(shape=input_shape, name='input_res1')
            if(self.efficientnet_variant = 0):
                efficientnet_module = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif(self.efficientnet_variant = 1):
                efficientnet_module = EfficientNetB1(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif(self.efficientnet_variant = 2):
                efficientnet_module = EfficientNetB2(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif(self.efficientnet_variant = 3):
                efficientnet_module = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            elif(self.efficientnet_variant = 4):
                efficientnet_module = EfficientNetB4(include_top=False, weights='imagenet', input_tensor=input_layer, pooling=None)
            if(self.block_variant = 'Block1to7'):
                self.efficientnet_features = {0: {'res1': 41, 'res2': 69, 'res3': 155, 'res4': 213}, 1: {'res1': 68, 'res2': 111, 'res3': 227, 'res4': 328}, 2: {'res1': 68, 'res2': 111, 'res3': 227, 'res4': 328}, 3: {'res1': 68, 'res2': 111, 'res3': 257, 'res4': 373}, 4: {'res1': 83, 'res2': 141, 'res3': 317, 'res4': 463}}.[self.efficientnet_variant]  
            else:
                self.efficientnet_features = {0: {'res1': 41, 'res2': 69, 'res3': 155, 'res4': 129}, 1: {'res1': 68, 'res2': 111, 'res3': 227, 'res4': 300}, 2: {'res1': 68, 'res2': 111, 'res3': 227, 'res4': 300}, 3: {'res1': 68, 'res2': 111, 'res3': 257, 'res4': 330}, 4: {'res1': 83, 'res2': 141, 'res3': 317, 'res4': 390}}.[self.efficientnet_variant]
        
        # Set block-variant parameter
        if(self.block_variant = 'Block1to5'):
            block_num = 3
            block_bridge = 2
        elif(self.block_variant = 'Block1to5b'):
            block_num = 3
            block_bridge = 3
        else:
            block_num = 4
            block_bridge = 3
            
        # EfficientHourglass architecture
        for block in range(0, block_num):
            
            # EfficientNet back-end
            Block = efficientnet_module.layers[self.efficientnet_features['res{0}'.format(block+1)]].output 
            map_numb = Block.shape.as_list()[-1]
            map_numb_block.append(map_numb)
            
            # Bridge MBconv blocks
            if(block < block_bridge):
                Bridge = self.MBconv(feature_maps=Block, filters=map_numb_block[block], name = 'Bridge1' + '_Res{}'.format(block+1), factor = MBconv_factor[block], se_ratio = SE_ratio[block], trainable = trainable)
                Bridge_block.append(Bridge)
        
        # Transpose upscaling
        for upsamp in range(1, block_num):
            Block = self.Transpose_concat_squeeze(feature_maps1 = Block, feature_maps2 = Bridge_block[Block_num-upsamp-1], filters = map_numb_block[Block_num-upsamp-1], se_ratio = SE_ratio[Block_num-upsamp-1], trainable = trainable)
        
        # Output and upscale stage for test
        if not self.upscale:
            Conf_maps = Conv2D(num_body_parts, 1, name='stage1_confs_tune', padding='same', kernel_initializer = CONV_KERNEL_INITIALIZER1)(Block)
        else:
            output_block = Conv2D(num_body_parts, 1, name='stage1_confs_tune', padding='same')(Block)
            deconvX = self.upscale_block(detection_layer=output_block, num_body_parts=num_body_parts)
            Conf_maps = deconvX
        self.model_outputs.append(Conf_maps) 
        
        # Build CIMA-Pose model
        self.model = Model(input_layer, self.model_outputs, name='EfficientHourglass')

    def Transpose_concat_squeeze(self, feature_maps1, feature_maps2, filters, se_ratio, trainable):
        Upsamp_BN = Conv2DTranspose(filters, kernel_size = (4, 4), strides = (2, 2), padding='same', trainable = trainable)(feature_maps1)
        Upsamp_BN = BatchNormalization()(Upsamp_BN)
        if(self.model_type = 'L'):
            Upsamp_BN = ReLU(max_value = 6)(Upsamp_BN)
        else:
            Upsamp_BN = Swish('swish1')(Upsamp_BN)
        output_block = Add()([feature_maps2, Upsamp_BN])
        if not(self.model_type = 'L')
            output_block = self.SE_EfficientNet(input_x = output_block, input_filters = output_block.shape.as_list()[-1], se_ratio = se_ratio, trainable = trainable)
        return output_block
        
    def MBconv(self, feature_maps, filters, name, factor, se_ratio, trainable):
        
        # Inverted bottelneck (expand features)
        if(factor==1):
            ConvBN1 = feature_maps
        else:
            ConvBN1 = Conv2D(factor*filters, (1, 1), name = name + '_BN1', padding='same', trainable = trainable)(feature_maps)
            ConvBN1 = BatchNormalization()(ConvBN1)
            if(self.model_type = 'L' or 'H'):
                ConvBN1 = ReLU(max_value = 6)(ConvBN1)
            else:
                ConvBN1 = Swish('swish1')(ConvBN1)
        
        # Depthwise convolutions
        dConv = DepthwiseConv2D(kernel_size = (5, 5), name = name + '_dConv', padding='same', trainable = trainable)(ConvBN1)
        dConv = BatchNormalization()(dConv)
        if(self.model_type = 'L'):
            output = ReLU(max_value = 6)(dConv)
        else:
            output = Swish('swish1')(dConv)
            input_filters = output.shape.as_list()[-1]
            output = self.SE_EfficientNet(input_x = dConv, input_filters = input_filters, se_ratio = se_ratio, trainable = trainable)
        
        # Bottelneck
        output = Conv2D(filters, (1, 1), name = name + '_BN2', padding='same', trainable = trainable)(output)
        output = BatchNormalization()(output)
        if(self.model_type = 'L'):
            output = ReLU(max_value = 6)(output)
        else:
            output = Swish('swish1')(output)
        return ConvBN2; 
    
    
    def SE_EfficientNet(self, input_x, input_filters, se_ratio, trainable):
        num_reduced_filters = max(
            1, int(input_filters / se_ratio))
        if K.image_data_format() == "channels_first":
            channel_axis = 1
            spatial_dims = [2, 3]
        else:
            channel_axis = -1
            spatial_dims = [1, 2]

        # Squeeze-and-Excitation  

        # Squeeze
        x = input_x
        x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                trainable = trainable,
                #kernel_initializer = kernel_initializer,
                padding='same',
                use_bias=True
            )(x)
        x = Swish('swish1')(x)

        # Excite
        x = Conv2D(
                input_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                trainable = trainable,
                #kernel_initializer = kernel_initializer,
                padding='same',
                use_bias=True
            )(x)
        x = Activation('sigmoid')(x)
        output_x = Multiply()([x, input_x])

        return output_x  
        
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
        Numpy array pre-processed according to EfficientNet (Lite) standard
    """
    if(self.model_type = 'L'):
        return efficientnet_lite.preprocess_input(x)
    else:
        return efficientnet_preprocess_input(x)
    
if __name__ == '__main__':
    efficienthourglass = architecture()
    efficienthourglass.model.summary()

