from tensorflow.keras import backend as K

def euclidean_loss(y_true, y_pred):
    """ Returns Sum-of-Squares/Euclidean Loss of the given tensors.

    Parameters
    ----------
    y_true: ndarray
        Tensor containing target values
    y_pred: ndarray
        Tensor containing predited values

    Returns
    ----------
    L: ndarray
        Tensor of loss values (the mean squared loss per pixel for each example)
    """
    
    size = K.shape(y_pred)
    height = size[1]
    width = size[2]
    num_body_parts = size[3]
    pixels = K.cast(height * width * num_body_parts, "float32")
    
    return K.sum(K.square(y_true - y_pred), axis=[1,2,3]) / pixels