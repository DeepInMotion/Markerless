import os
import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import make_confidence_maps, make_part_affinity_fields, make_body_segments, pad, resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from utils.image import ImageDataGenerator
import math
import random
import copy

import warnings
warnings.filterwarnings("ignore", message="The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.")
warnings.filterwarnings("ignore", message="(NumpyArrayIterator)(.*)")

class DataGenerator:
    
    def __init__(self, df, settings):
        
        # Dataframe
        self.df = df
        
        # Settings
        input_size = settings['input_size']
        output_size = settings['output_size']
        batch_size = settings['batch_size']
        aug_rotation = settings['aug_rotation']
        aug_zoom = settings['aug_zoom']
        aug_flip = settings['aug_flip']
        body_parts = settings['body_parts']
        flipped_body_parts = settings['flipped_body_parts']
        
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        
        self.output_height = output_size[0]
        self.output_width = output_size[1]
        
        self.batch_size = batch_size
        
        # Data augmentation configurations 
        data_gen_args = dict(rotation_range=aug_rotation, 
                             width_shift_range=0.0,
                             height_shift_range=0.0,
                             zoom_range=aug_zoom,
                             horizontal_flip=aug_flip,
                             fill_mode='constant',
                             cval=0.)
        
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.template_datagen = ImageDataGenerator(**data_gen_args)
        self.point_datagen = ImageDataGenerator(**data_gen_args, flipped_body_parts=flipped_body_parts, body_parts=body_parts)
        
    def n_steps(self):
        return math.ceil(self.df.shape[0] / self.batch_size)  
    
    def load_one_hot_point(self, id):
        row = self.df.loc[id]
        points = row.points
        
        input_points = np.asarray([(int(p[0] * self.output_width), int(p[1] * self.output_height)) for p in points], dtype=np.uint32)
        
        # Create 1-hot point tensor
        n_points = len(input_points)
        one_hot_points = np.zeros((self.output_height, self.output_width, n_points), dtype='bool')
        
        for i, (p_x, p_y) in enumerate(input_points):
            if np.any([p_x, p_y]):
                if (0 <= p_x <= self.output_width) and (0 <= p_y <= self.output_height):
                    
                    if p_x == self.output_width:
                        p_x -= 1
                    if p_y == self.output_height:
                        p_y -= 1
                        
                    one_hot_points[p_y, p_x, i] = True
                
        return one_hot_points
    
    def load_img(self, id, color_jitter=False):
        row = self.df.loc[id]
        path = row.img_path
        
        img  = load_img(path)
        if color_jitter:
            jitter_channel = random.randint(0,2)
            colored_array = img_to_array(img) / 255.
            colored_array[...,jitter_channel] += random.uniform(-0.25,0.25)
            img = array_to_img(colored_array)
        
        if not img.height == img.width:
            padded_array = pad(img_to_array(img) / 255., self.input_height, self.input_width)
            img = array_to_img(padded_array)
            
        if img.height != self.input_height or img.width != self.input_width:
            resized_array = resize(img_to_array(img) / 255., self.input_height, self.input_width)
            img = array_to_img(resized_array)
        
        img = img_to_array(img)
            
        return img
    
    def labels(self, confs, pafs=None, body_segments=None, model_type='CP-CIMA'):
        
        # One sigma value
        conf = confs[0]
        if pafs is not None:
            paf = pafs[0]
        if body_segments is not None:
            segment = body_segments[0]
        if model_type.upper() == 'CP-2-TUNE': # CIMA-Pose: Training with 2 passes
            return {
                    'stage1_confs_tune': conf,
                    'stage2_confs_tune': conf
                   }
        elif model_type.upper() == 'EH-1-TUNE': # EfficientHourglass: Training with 1 pass
            return {
                    'stage1_confs_tune': conf
                    }
        elif model_type.upper() == 'EH-1': # EfficientHourglass: Training with 1 pass
            return {
                    'stage1_confs': conf
                    }
        elif model_type.upper() == 'EP-1+2-PAFS-TUNE': # EfficientPose with 1PAFs+2 passes
            return {
                    'pass1_skeleton_pafs_tune': paf,
                    'pass2_detection1_confs_tune': conf,
                    'pass3_detection2_confs_tune': conf
                   }
    
    def load_points(self, id):
        row = self.df.loc[id]
        points = row.points
        
        return points
    
    def one_hot_to_points(self, one_hot):
        """Extract coordinate list from a 3d-matrix of shape (h, w, c)"""
        
        arr = one_hot.transpose(2,0,1)
        abs_y_x = [np.unravel_index(channel.argmax(), channel.shape) for channel in arr]
        rel_x_y = [(x / self.output_width, y / self.output_height) for (y, x) in abs_y_x]
        
        return np.asarray(rel_x_y)
    
    def get_pred_data(self, batch_size, n_samples=None, preprocess_input=None, supply_ids=False):
        
        ids = self.df.index.to_numpy()
        if n_samples:
            selection = np.random.choice(ids, n_samples, replace=False)
        else:
            n_samples = len(ids)
            selection = ids
            
        for ndx in range(0, n_samples, batch_size):
            b_selection = selection[ndx:min(ndx + batch_size, n_samples)]
            b_size = len(b_selection)

            imgs_batch = np.asarray([self.load_img(sample_id) for sample_id in b_selection])
            if preprocess_input:
                imgs_batch = preprocess_input(imgs_batch)
            
            if supply_ids:
                yield b_selection, imgs_batch
            else:
                yield imgs_batch
    
    def get_eval_data(self, batch_size, n_samples=None, preprocess_input=None, supply_ids=False, shuffle=False):
        
        ids = self.df.index.to_numpy()
        if n_samples:
            selection = np.random.choice(ids, n_samples, replace=False)
        else:
            n_samples = len(ids)
            selection = ids
        
        if shuffle:
            np.random.shuffle(selection)
            
        for ndx in range(0, n_samples, batch_size):
            b_selection = selection[ndx:min(ndx + batch_size, n_samples)]
            b_size = len(b_selection)

            imgs_batch = np.asarray([self.load_img(sample_id) for sample_id in b_selection])
            if preprocess_input:
                imgs_batch = preprocess_input(imgs_batch)
                
            points_batch = [self.load_points(sample_id) for sample_id in b_selection]

            if supply_ids:
                yield b_selection, imgs_batch, points_batch
            else:
                yield imgs_batch, points_batch
    
    def get_data(self, batch_size, segments, model_type, n_samples=None, schedule=None, sigma=None, initial_epoch=0, augment=True, shuffle=True, segment_width=None, supply_pafs=False, supply_body_segments=False, supply_points=False, num_resolutions=1, preprocess_input=None, detection_sizes=None, epochs_per_dataset=None, color_jitter=False):
        
        ids = self.df.index.to_numpy()

        if n_samples:
            selection = np.random.choice(ids, n_samples, replace=False)
        else:
            n_samples = len(ids)
            selection = ids
                    
        # Make generator infinite
        epoch = initial_epoch
        if epochs_per_dataset is not None:
            full_selection = copy.deepcopy(selection)
            portion_size = math.ceil(n_samples / epochs_per_dataset)
            random.shuffle(full_selection)
            dataset_portions = [full_selection[i*portion_size:portion_size+i*portion_size] for i in range(epochs_per_dataset)]
            portion_index = epoch % epochs_per_dataset#0

        while True:
            
            if epochs_per_dataset is not None:
                selection = dataset_portions[portion_index]
                epoch_size = len(selection)
                portion_index += 1
                if portion_index >= epochs_per_dataset:
                    portion_index = 0
            else:
                epoch_size = n_samples
                
            # obtain sigma according to epoch
            joint_sigma = len(schedule[0]) == 2
            if schedule is not None:
                if joint_sigma:
                    pass
                else:    
                    for (interval_sigma, interval_segment_width, start_epoch, l_rate) in schedule:
                        if epoch >= start_epoch: 
                            sigma = interval_sigma
                            segment_width = interval_segment_width
                        else:
                            break
            
            if shuffle:
                np.random.shuffle(selection)
                
            for ndx in range(0, epoch_size, batch_size):
                
                b_selection = selection[ndx:min(ndx + batch_size, epoch_size)]
                b_size = len(b_selection)
                
                imgs_batch = np.asarray([self.load_img(sample_id, color_jitter=color_jitter) for sample_id in b_selection])
                if preprocess_input:
                    imgs_batch = preprocess_input(imgs_batch)

                if augment:                    
                    flow_args = {'batch_size': b_size, 'shuffle': False, 'seed': np.random.randint(0, 100)}
                    
                    one_hot_points_batch = np.asarray([self.load_one_hot_point(sample_id) for sample_id in b_selection])

                    imgs_batch_aug = self.image_datagen.flow(imgs_batch, **flow_args).next()
                    one_hot_points_batch_aug = self.point_datagen.flow(one_hot_points_batch, **flow_args).next()
                    points_batch_aug =[self.one_hot_to_points(one_hot_points) for one_hot_points in one_hot_points_batch_aug]
                    
                    if joint_sigma:
                        confs_batch_aug = []
                        for s in sigma:
                            confs_batch_aug.append(np.asarray([make_confidence_maps(points_aug, sigma=s, height=self.output_height, width=self.output_width) for points_aug in points_batch_aug]))
                    elif num_resolutions > 1:
                        confs_batch_aug = []
                        for i in range(num_resolutions):
                            s = max(2.0, sigma / (2**i)) 
                            height = self.output_height // (2**i)
                            width = self.output_width // (2**i)
                            confs_batch_aug.append(np.asarray([make_confidence_maps(points_aug, sigma=s, height=height, width=width) for points_aug in points_batch_aug])) 
                    elif not detection_sizes is None:
                        confs_batch_aug = []
                        reference_size = detection_sizes[0]
                        for detection_size in detection_sizes:
                            size_factor = detection_size / reference_size
                            s = max(1.3, sigma * size_factor) 
                            height = detection_size
                            width = detection_size
                            confs_batch_aug.append(np.asarray([make_confidence_maps(points_aug, sigma=s, height=height, width=width) for points_aug in points_batch_aug])) 
                    else:
                        confs_batch_aug = [np.asarray([make_confidence_maps(points_aug, sigma=sigma, height=self.output_height, width=self.output_width) for points_aug in points_batch_aug])]
                    if supply_pafs:
                        if num_resolutions > 1:
                            pafs_batch_aug = []
                            for i in range(num_resolutions):
                                s = max(1.0, segment_width / (2**i)) 
                                height = self.output_height // (2**i)
                                width = self.output_width // (2**i)
                                pafs_batch_aug.append(np.asarray([make_part_affinity_fields(points_aug, segment_width=s, height=height, width=width) for points_aug in points_batch_aug])) 
                        else:
                            pafs_batch_aug = [np.asarray([make_part_affinity_fields(points_aug, segments=segments, segment_width=segment_width, height=self.output_height, width=self.output_width) for points_aug in points_batch_aug])]
                        labels_batch_aug = self.labels(confs_batch_aug, pafs=pafs_batch_aug, model_type=model_type)
                    elif supply_body_segments:
                        if num_resolutions > 1:
                            body_segments_batch_aug = []
                            for i in range(num_resolutions):
                                s = max(1.0, segment_width / (2**i)) 
                                height = self.output_height // (2**i)
                                width = self.output_width // (2**i)
                                body_segments_batch_aug.append(np.asarray([make_body_segments(points_aug, segment_width=s, height=height, width=width) for points_aug in points_batch_aug])) 
                        else:
                            body_segments_batch_aug = [np.asarray([make_body_segments(points_aug, segments=segments, segment_width=segment_width, height=self.output_height, width=self.output_width) for points_aug in points_batch_aug])]
                        labels_batch_aug = self.labels(confs_batch_aug, body_segments=body_segments_batch_aug, model_type=model_type)
                    else:
                        labels_batch_aug = self.labels(confs_batch_aug, model_type=model_type)
                
                    if supply_points:
                        yield imgs_batch_aug, labels_batch_aug, points_batch_aug
                    else:
                        yield imgs_batch_aug, labels_batch_aug
                
                else:
                    points_batch = [self.load_points(sample_id) for sample_id in b_selection]
                    
                    if joint_sigma:
                        confs_batch = []
                        for s in sigma:
                            confs_batch.append(np.asarray([make_confidence_maps(points, sigma=s, height=self.output_height, width=self.output_width) for points in points_batch]))    
                    elif num_resolutions > 1:
                        confs_batch = []
                        for i in range(num_resolutions):
                            s = max(1.3, sigma / (2**i)) 
                            height = self.output_height // (2**i)
                            width = self.output_width // (2**i)
                            confs_batch.append(np.asarray([make_confidence_maps(points, sigma=s, height=height, width=width) for points in points_batch])) 
                    elif not detection_sizes is None:
                        confs_batch = []
                        reference_size = detection_sizes[0]
                        for detection_size in detection_sizes:
                            size_factor = detection_size / reference_size
                            s = max(1.3, sigma * size_factor) 
                            height = detection_size
                            width = detection_size
                            confs_batch.append(np.asarray([make_confidence_maps(points, sigma=s, height=height, width=width) for points in points_batch])) 
                    else:
                        confs_batch = [np.asarray([make_confidence_maps(points, sigma=sigma, height=self.output_height, width=self.output_width) for points in points_batch])]
                        
                    if supply_pafs:
                        if num_resolutions > 1:
                            pafs_batch = []
                            for i in range(num_resolutions):
                                s = max(1.0, segment_width / (2**i)) 
                                height = self.output_height // (2**i)
                                width = self.output_width // (2**i)
                                pafs_batch.append(np.asarray([make_part_affinity_fields(points, segment_width=s, height=height, width=width) for points in points_batch])) 
                        else:
                            pafs_batch = [np.asarray([make_part_affinity_fields(points, segments=segments, segment_width=segment_width, height=self.output_height, width=self.output_width) for points in points_batch])]
                        labels_batch = self.labels(confs_batch, pafs=pafs_batch, model_type=model_type)
                    elif supply_body_segments:
                        if num_resolutions > 1:
                            body_segments_batch = []
                            for i in range(num_resolutions):
                                s = max(1.0, segment_width / (2**i)) 
                                height = self.output_height // (2**i)
                                width = self.output_width // (2**i)
                                body_segments_batch.append(np.asarray([make_body_segments(points, segment_width=s, height=height, width=width) for points in points_batch])) 
                        else:
                            body_segments_batch = [np.asarray([make_body_segments(points, segments=segments, segment_width=segment_width, height=self.output_height, width=self.output_width) for points in points_batch])]
                        labels_batch = self.labels(confs_batch, body_segments=body_segments_batch, model_type=model_type)
                    else:
                        labels_batch = self.labels(confs_batch, model_type=model_type)
                    
                    if supply_points:
                        yield imgs_batch, labels_batch, points_batch
                    else:
                        yield imgs_batch, labels_batch
                
            epoch += 1