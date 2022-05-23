import numpy as np
import skimage.transform
import skimage.util
import skimage.io
from PIL import ImageDraw
import matplotlib.pyplot as plt
import os
from scipy.ndimage.filters import gaussian_filter
from scipy import linalg
from numpy import random
import math
random.seed(42)

def confidence(p1_x, p1_y, p0_x, p0_y, sigma):
    return math.exp(-(linalg.norm(np.asarray([p1_x,p1_y])-np.asarray([p0_x,p0_y]), ord=2)**2)/(sigma**2))

def make_conf_map(p0_x, p0_y, height, width, sigma):
    if p0_x == 0 and p0_y == 0:
        return np.zeros((height, width))
    
    xs = np.linspace(0, width, width, endpoint=False, dtype=int)
    ys = np.linspace(0, height, height, endpoint=False, dtype=int)
    
    return confidence(xs[None, :], ys[:, None], p0_x, p0_y, sigma=sigma)

def make_confidence_maps(points,  height, width, sigma):
    """ Generate confidence maps for each coordinate in the coordinate list.
    
    Parameters
    ----------
    points: list[(int, int)]
        List containing (x, y) coordinates for body parts 
    height: int
        The wanted height of the part affinity fields
    width: int
        The wanted width of the part affinity fields
    sigma: float
        Sigma value to use in confidence maps
        
    Returns
    ----------
    C: ndarray
        Tensor of shape HxWxC with confidence maps distributed along its third axis.
    """
    
    abs_points = [(int(p_x * width), int(p_y * height)) for (p_x, p_y) in points]
    conf_maps = [make_conf_map(p_x, p_y, height, width, sigma=sigma) for (p_x, p_y) in abs_points]

    return np.dstack(conf_maps)

def show_confidence_maps(confidence_maps, img):
    conf_size = confidence_maps.shape[:2]
    if img.shape != conf_size:
        img = pad(img, conf_size[0], conf_size[1])
    
    plt.figure()
    plt.imshow(img)
    plt.imshow(np.max(confidence_maps, axis=2), alpha=0.8)
    plt.show()
    
def segment_unit_vector(p0_abs, p1_abs):
    """ Calculate the unit vector pointing along the segment between x0 and x1
    
    Parameters
    ----------
    p0_abs: array[int, int]
        Starting point of segment, in absolute coordinates
    p1_abs: array[int, int]
        Stop point of segment, in absolute coordinates
        
    Returns
    ----------
    v: array[float, float]
        Unit vector pointing along the segment between x0 to x1
    """
    
    l2_norm = np.linalg.norm(p1_abs - p0_abs, ord=2)
    
    if l2_norm:
        v = np.divide(p1_abs - p0_abs, l2_norm)
    else:
        v = np.array([0.0, 0.0])
        
    return v

def segment_length(p0_abs, p1_abs):
    """Calculate the length between points p0 and p1 
    
    Parameters
    ----------
    p0_abs: array[int, int]
        Starting point of segment, in absolute coordinates
    p1_abs: array[int, int]
        Stop point of segment, in absolute coordinates
        
    Returns
    ----------
    length: float
        The length of the segment between p0 and p1
    """
    
    length = np.linalg.norm(np.subtract(p1_abs, p0_abs), ord=2)
    
    return length

def perpendicular_vector(v):
    """ Calculate a vector which is perpendicular to v
    Parameters
    ----------
    v: array[float, float]
        A 2d vector
    
    Returns
    ----------
    v_perp: array[float, float]
        A 2d vector perpendicular to v
    
    """
    
    v_perp = np.array([v[1], -v[0]])
    
    return v_perp

def d1(p_x, p_y, p0_x, p0_y, v):
    return np.dot(v, np.subtract([p_x, p_y], [p0_x, p0_y]))

def d2(p_x, p_y, p0_x, p0_y, v):
    v_perp = perpendicular_vector(v)
    return np.dot(v_perp, np.subtract([p_x, p_y], [p0_x, p0_y]))

def make_paf(p0_x, p0_y, p1_x, p1_y, height, width, segment_width):
    p0 = np.array([p0_x, p0_y])
    p1 = np.array([p1_x, p1_y])
    
    paf = np.zeros((height, width, 2))
    
    if not (np.any(p0) and np.any(p1)):
        return paf
    
    v = segment_unit_vector(p0, p1)
    l = segment_length(p0, p1)
        
    xs = np.linspace(0, width, width, endpoint=False, dtype=int)
    ys = np.linspace(0, height, height, endpoint=False, dtype=int)
    
    D1 = d1(xs[None, :], ys[:, None], p0_x, p0_y, v)
    D2 = d2(xs[None, :], ys[:, None], p0_x, p0_y, v)
    
    A = np.logical_and(np.greater_equal(D1, 0), np.less_equal(D1, l))
    B = np.less_equal(np.absolute(D2), segment_width)
    point_on_segment = np.logical_and(A, B)
    
    paf[point_on_segment] = v
    
    return paf

def make_part_affinity_fields(points, segments, height, width, segment_width):
    """ Generate part affinity field for each segment in the segments list.
    
    Parameters
    ----------
    points: list[(float, float)]
        List of (x, y) tuples, where x and y are relative coordinates.
    segments: list[(int, int)]
        List of (i1, i2) tuples, where each tuple connects two points to a segment. 
    height: int
        The wanted height of the part affinity fields
    width: int
        The wanted width of the part affinity fields
    segment_width: float
        Segment width
        
    Returns
    ----------
    PAFS: ndarray (HxWx2L)
        L Part affinity fields of shape HxWx2, stacked along the 2nd axis
    """
    
    p0s = [points[l0] for (l0, l1) in segments]
    p1s = [points[l1] for (l0, l1) in segments]
    
    abs_p0s = [(int(p_x * width), int(p_y * height)) for (p_x, p_y) in p0s]
    abs_p1s = [(int(p_x * width), int(p_y * height)) for (p_x, p_y) in p1s]
    
    pafs = [make_paf(*p0, *p1, height=height, width=width, segment_width=segment_width) for (p0, p1) in zip(abs_p0s, abs_p1s)]
    
    return np.dstack(pafs)
        
def show_part_affinity_fields(part_affinity_fields, img):
    plt.figure()
    plt.imshow(img)
    for i in range(0, part_affinity_fields.shape[-1], 2):
        U = part_affinity_fields[..., i]
        V = part_affinity_fields[..., i+1]

        if not (np.any(U.flatten()) or np.any(V.flatten())):
                continue

        plt.quiver(U, V, angles='xy', scale=6, scale_units='inches', minlength=0.01, color='r', pivot='mid')
    plt.show()
    
def make_body_segment(p0_x, p0_y, p1_x, p1_y, height, width, segment_width):
    p0 = np.array([p0_x, p0_y])
    p1 = np.array([p1_x, p1_y])
    
    body_segment = np.zeros((height, width, 1))
    
    if not (np.any(p0) and np.any(p1)):
        return body_segment
    
    v = segment_unit_vector(p0, p1)
    l = segment_length(p0, p1)
        
    xs = np.linspace(0, width, width, endpoint=False, dtype=int)
    ys = np.linspace(0, height, height, endpoint=False, dtype=int)
    
    D1 = d1(xs[None, :], ys[:, None], p0_x, p0_y, v)
    D2 = d2(xs[None, :], ys[:, None], p0_x, p0_y, v)
    
    A = np.logical_and(np.greater_equal(D1, 0), np.less_equal(D1, l))
    B = np.less_equal(np.absolute(D2), segment_width)
    point_on_segment = np.logical_and(A, B)
    
    body_segment[point_on_segment] = 1.0
    
    return body_segment
    
def make_body_segments(points, height, width, segment_width, segments):
    """ Generate body segment for each segment in the segments list.
    
    Parameters
    ----------
    points: list[(float, float)]
        List of (x, y) tuples, where x and y are relative coordinates.
    segments: list[(int, int)]
        List of (i1, i2) tuples, where each tuple connects two points to a segment. 
    height: int
        The wanted height of the whole map
    width: int
        The wanted width of the whole map
    segment_width: float
        Segment width
        
    Returns
    ----------
    body_segments: ndarray (HxWxL)
        L Body Segments of shape HxWx1, stacked along the 2nd axis
    """
    
    p0s = [points[l0] for (l0, l1) in segments]
    p1s = [points[l1] for (l0, l1) in segments]
    
    abs_p0s = [(int(p_x * width), int(p_y * height)) for (p_x, p_y) in p0s]
    abs_p1s = [(int(p_x * width), int(p_y * height)) for (p_x, p_y) in p1s]
    
    body_segments = [make_body_segment(*p0, *p1, height=height, width=width, segment_width=segment_width) for (p0, p1) in zip(abs_p0s, abs_p1s)]
    
    return np.dstack(body_segments)

def show_body_segments(body_segments, img):
    map_size = body_segments.shape[:2]
    if img.shape != map_size:
        img = pad(img, map_size[0], map_size[1])
    
    plt.figure()
    plt.imshow(img)
    plt.imshow(np.max(body_segments, axis=2), alpha=0.8)
    plt.show()
    
def resize(source_array, target_height, target_width):
    """ Resizes an image or image-like ndarray to be no larger than (target_height, target_width) or (target_height, target_width, c)
    
    Parameters
    ----------
    source_array: ndarray
        Imagelike numpy ndarray of shape (h, w) or (h, w, c)
    target_height: int
        Height of the padded image
    target_width: int
        Width of the padded image
    Returns
    ----------
    resized_array: ndarray
        Resized ndarray with size *no larger than* (target_height, target_width) or (target_height, target_width, c)
    """
    source_height, source_width = source_array.shape[:2]
            
    target_ratio = target_height / target_width
    source_ratio = source_height / source_width

    if target_ratio > source_ratio:
        scale = target_width / source_width
    else:
        scale = target_height / source_height

    resized_array = skimage.transform.rescale(source_array, scale, multichannel=True)
    
    return resized_array

def pad(source_array, target_height, target_width):
    """ Pads an image or image-like ndarray with zeros, to fit the target-size.
    
    Parameters
    ----------
    source_array: ndarray
        Imagelike numpy ndarray of shape (h, w) or (h, w, c)
    target_height: int
        Height of the padded image
    target_width: int
        Width of the padded image
    
    Returns
    ----------
    target_array: ndarray
        Zero-padded image-like numpy ndarray of shape (target_height, target_width) or (target_height, target_width, c)
    """
    
    source_height, source_width = source_array.shape[:2]
    
    if (source_height > target_height) or (source_width > target_width):
        source_array = resize(source_array, target_height, target_width)
        source_height, source_width = source_array.shape[:2]
        
    pad_left = int((target_width - source_width) / 2)
    pad_top = int((target_height - source_height) / 2)
    pad_right = int(target_width - source_width - pad_left)
    pad_bottom = int(target_height - source_height - pad_top)
    
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    
    has_channels_dim = len(source_array.shape) == 3
    if has_channels_dim:  
        paddings.append([0,0])
        
    target_array = skimage.util.pad(source_array, paddings, 'constant')
    
    return target_array

def extract_point(conf, threshold, confidence=False):
    # conf: 2d confidence map of shape (height, width)
    
    output_width = conf.shape[1]
    output_height = conf.shape[0]
    
    # Apply gaussian filter
    gauss = gaussian_filter(conf, sigma=1.)
        
    # Locate peaks in confidence map
    max_index = np.argmax(gauss)
    peak_y = float(math.floor(max_index / output_width))
    peak_x = max_index % output_width
    
    # Verify confidence of prediction
    conf_value = gauss[int(peak_y),int(peak_x)]
    if conf_value < threshold:
        peak_x = 0.0
        peak_y = 0.0
    else:
        peak_x += 0.5
        peak_y += 0.5
                
    peak_x /= output_width
    peak_y /= output_height
    
    if confidence:
        return peak_x, peak_y, conf_value
    else:
        return peak_x, peak_y
