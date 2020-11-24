from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input as _preprocess_input

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

class efficientnet_lite:
    def __init__(self, variant=0, input_resolution=None):
        """ Initializes EfficientNet Lite model. """
        
        # Variables
        self.variant = variant #[0,1,2,3,4]
                        
        # Load EfficientNet Lite model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'L{0}.h5'.format(variant))
        self.model = keras.models.load_model(model_path)
        
        # Set input resolution
        if input_resolution is not None:
            self.model._layers[0]._batch_input_shape = (None, input_resolution, input_resolution, 3)
            self.model = keras.models.model_from_json(self.model.to_json())
    
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
    
    return _preprocess_input(x, mode='tf')

if __name__ == '__main__':
    efficientnet_lite = efficientnet_lite()
    keras.utils.print_summary(efficientnet_lite.model, line_length=200)
