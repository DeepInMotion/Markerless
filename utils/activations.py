from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation 
from tensorflow.keras.utils import get_custom_objects

class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'
    
def swish1(x):
    return x * K.sigmoid(x)

def eswish(x):
    beta = 1.25
    return beta * x * K.sigmoid(x)

get_custom_objects().update({'Swish': Swish(eswish), 'eswish': eswish, 'swish1': swish1})