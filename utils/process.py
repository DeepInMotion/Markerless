import os
import random
from PIL import Image
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.helpers import pad, resize

def copy_images(image_names, from_dir, to_dir):

    # Copy images to new directory
    os.makedirs(to_dir, exist_ok=True)
    for image_name in tqdm(image_names):
        image = Image.open(os.path.join(from_dir, image_name))
        image.save(os.path.join(to_dir, image_name))

def generate_datasets(raw_images_dir, trainval_test_split, train_val_split):

    # Find image files
    raw_file_names = os.listdir(raw_images_dir)
    raw_image_names = []
    for raw_file_name in raw_file_names:
        if raw_file_name.lower().endswith(('.png', '.jpg')):
            raw_image_names.append(raw_file_name)  

    # Shuffle images to create pseudo-random datasets
    random.seed(1433)
    random.shuffle(raw_image_names)
    
    # Determine trainval and test split
    num_images = len(raw_image_names)
    num_trainval_images = int(num_images * trainval_test_split)
    trainval_image_names = raw_image_names[:num_trainval_images]
    test_image_names = raw_image_names[num_trainval_images:]
    
    # Determine train and validation split
    num_train_images = int(num_trainval_images * train_val_split)
    train_image_names = trainval_image_names[:num_train_images]
    validation_image_names = trainval_image_names[num_train_images:]
    
    # Generate raw image folders for datasets
    print("-- Generate train data")
    copy_images(train_image_names, raw_images_dir, os.path.join(raw_images_dir, 'train'))
    print("-- Generate val data")
    copy_images(validation_image_names, raw_images_dir, os.path.join(raw_images_dir, 'val'))
    print("-- Generate test data")
    copy_images(test_image_names, raw_images_dir, os.path.join(raw_images_dir, 'test'))
    
def perform_processing(project_constants):
    
    # Store annotations in dictionary
    raw_annotations = {}
    with open(project_constants.RAW_ANNOTATION_FILE, 'r') as annotations_csv:
        annotations = csv.reader(annotations_csv)
        header = next(annotations)
        for annotation in annotations:
            try:
                image_file_path = annotation[1]
                if '\\' in image_file_path:
                    image_file_name = image_file_path.split('\\')[1]
                elif '/' in image_file_path:
                    image_file_name = image_file_path.split('/')[1]
                else:
                    image_file_name = image_file_path
                image_coordinates = annotation[2:]
                if len(image_coordinates[0]) > 0:
                    raw_annotations[image_file_name] = image_coordinates
            except:
                image_file_path = None
    
    # Make processed images and corresponding labels
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        dataset_raw_images_dir = os.path.join(project_constants.RAW_IMAGES_DIR, dataset)
        if os.path.exists(dataset_raw_images_dir):
            
            # Fetch dataset images
            dataset_file_names = os.listdir(dataset_raw_images_dir)
            dataset_image_names = []
            for dataset_file_name in dataset_file_names:
                if dataset_file_name.lower().endswith(('.png', '.jpg')):
                    dataset_image_names.append(dataset_file_name) 
                    
            # Create directories for processed images and labels
            dataset_processed_dir = os.path.join(project_constants.PROCESSED_DATA_DIR, dataset)
            dataset_processed_images_dir = os.path.join(dataset_processed_dir, 'images_{0}x{0}'.format(project_constants.MAXIMUM_RESOLUTION))
            dataset_processed_labels_dir = os.path.join(dataset_processed_dir, 'points')
            os.makedirs(dataset_processed_images_dir)
            os.makedirs(dataset_processed_labels_dir)
            
            # Process images and labels
            print('-- Process {0} images and labels'.format(dataset))
            for dataset_image_name in tqdm(dataset_image_names):
                if dataset_image_name in raw_annotations.keys():
                    
                    # Fetch raw image
                    raw_image = Image.open(os.path.join(dataset_raw_images_dir, dataset_image_name))
                    raw_image_width, raw_image_height = raw_image.size
                    
                    # Fetch raw annotation
                    raw_annotation = raw_annotations[dataset_image_name]
                    
                    # Crop images based on person ROI
                    if project_constants.CROP:
                
                        # Fetch minimum and maximum x and y values of labels
                        min_x = 1.0
                        max_x = 0.0
                        min_y = 1.0
                        max_y = 0.0
                        for body_part_coordinates in raw_annotation:
                            body_part_x, body_part_y = tuple(map(float, body_part_coordinates[1:-1].split(',')))
                            if body_part_x < min_x:
                                min_x = body_part_x
                            if body_part_x > max_x:
                                max_x = body_part_x
                            if body_part_y < min_y:
                                min_y = body_part_y
                            if body_part_y > max_y:
                                max_y = body_part_y
                        x_interval = max_x - min_x
                        y_interval = max_y - min_y
                        
                        # Determine ROI with desired padding
                        roi_min_x = min_x - project_constants.CROP_PADDING*x_interval
                        roi_max_x = max_x + project_constants.CROP_PADDING*x_interval
                        roi_min_y = min_y - project_constants.CROP_PADDING*y_interval
                        roi_max_y = max_y + project_constants.CROP_PADDING*y_interval
                        
                        # Expand ROI in case of difference between height and width
                        raw_roi_min_x = roi_min_x*raw_image_width
                        raw_roi_max_x = roi_max_x*raw_image_width
                        raw_roi_min_y = roi_min_y*raw_image_height
                        raw_roi_max_y = roi_max_y*raw_image_height
                        raw_roi_x_interval = raw_roi_max_x - raw_roi_min_x
                        raw_roi_y_interval = raw_roi_max_y - raw_roi_min_y
                        if raw_roi_x_interval > raw_roi_y_interval:  
                            difference = raw_roi_x_interval - raw_roi_y_interval#(raw_roi_x_interval - raw_roi_y_interval) / raw_image_height if wider else (raw_roi_x_interval - raw_roi_y_interval) / raw_image_width
                            raw_roi_min_x = raw_roi_min_x if raw_roi_min_x >= 0.0 else 0.0
                            raw_roi_max_x = raw_roi_max_x if raw_roi_max_x <= raw_image_width else raw_image_width
                            raw_roi_min_y = raw_roi_min_y - (difference / 2) if raw_roi_min_y - (difference / 2) >= 0.0 else 0.0
                            raw_roi_max_y = raw_roi_max_y + (difference / 2) if raw_roi_max_y + (difference / 2) <= raw_image_height else raw_image_height
                        else:
                            difference = raw_roi_y_interval - raw_roi_x_interval#(raw_roi_y_interval - raw_roi_x_interval) / raw_image_height if wider else (raw_roi_y_interval - raw_roi_x_interval) / raw_image_width
                            raw_roi_min_x = raw_roi_min_x - (difference / 2) if raw_roi_min_x - (difference / 2) >= 0.0 else 0.0
                            raw_roi_max_x = raw_roi_max_x + (difference / 2) if raw_roi_max_x + (difference / 2) <= raw_image_width else raw_image_width
                            raw_roi_min_y = raw_roi_min_y if raw_roi_min_y >= 0.0 else 0.0
                            raw_roi_max_y = raw_roi_max_y if raw_roi_max_y <= raw_image_height else raw_image_height
                            
                        # Crop image according to ROI
                        roi_min_x = raw_roi_min_x/raw_image_width
                        roi_max_x = raw_roi_max_x/raw_image_width
                        roi_min_y = raw_roi_min_y/raw_image_height
                        roi_max_y = raw_roi_max_y/raw_image_height
                        cropped_image = np.array(raw_image)[int(raw_roi_min_y):int(raw_roi_max_y), int(raw_roi_min_x):int(raw_roi_max_x)]
                        
                        # Make processed cropped image
                        resized_cropped_image = resize(cropped_image, project_constants.MAXIMUM_RESOLUTION, project_constants.MAXIMUM_RESOLUTION)
                        pad_cropped_image = pad(resized_cropped_image, project_constants.MAXIMUM_RESOLUTION, project_constants.MAXIMUM_RESOLUTION) 
                        processed_image = Image.fromarray(np.uint8(pad_cropped_image*255))
                        
                        # Transform body part coordinates according to ROI and padding
                        transformed_points = []
                        for raw_body_part_coordinates in raw_annotation:
                            
                            # Fetch raw coordinates
                            raw_body_part_x, raw_body_part_y = tuple(map(float, raw_body_part_coordinates[1:-1].split(',')))    
                            
                            # Transform according to ROI
                            roi_body_part_x = (raw_body_part_x - roi_min_x) / (roi_max_x - roi_min_x)
                            roi_body_part_y = (raw_body_part_y - roi_min_y) / (roi_max_y - roi_min_y)

                            # Transform according to padding
                            if pad_cropped_image.shape[1] > resized_cropped_image.shape[1]:
                                transformed_body_part_x = ((resized_cropped_image.shape[1] * roi_body_part_x) + (float(pad_cropped_image.shape[1] - resized_cropped_image.shape[1]) / 2)) / pad_cropped_image.shape[1]
                                transformed_body_part_y = roi_body_part_y                                
                                
                            elif pad_cropped_image.shape[0] > resized_cropped_image.shape[0]:   
                                transformed_body_part_x = roi_body_part_x
                                transformed_body_part_y = ((resized_cropped_image.shape[0] * roi_body_part_y) + (float(pad_cropped_image.shape[0] - resized_cropped_image.shape[0]) / 2)) / pad_cropped_image.shape[0]
                            else:
                                transformed_body_part_x = roi_body_part_x
                                transformed_body_part_y = roi_body_part_y
                            
                            # Bound coordinates between 0 and 1
                            if transformed_body_part_x < 0.0:
                                transformed_body_part_x = 0.0
                            elif transformed_body_part_x > 1.0:
                                transformed_body_part_x = 1.0
                            if transformed_body_part_y < 0.0:
                                transformed_body_part_y = 0.0
                            elif transformed_body_part_y > 1.0:
                                transformed_body_part_y = 1.0

                            transformed_points.append([transformed_body_part_x, transformed_body_part_y])

                        # Make processed transformed coordinates
                        processed_points = np.asarray(transformed_points)
                                                
                    # Utilize entire raw image
                    else:
                    
                        # Make processed image
                        processed_image = Image.fromarray(np.uint8(pad(resize(np.asarray(raw_image), project_constants.MAXIMUM_RESOLUTION, project_constants.MAXIMUM_RESOLUTION), project_constants.MAXIMUM_RESOLUTION, project_constants.MAXIMUM_RESOLUTION)*255))
                        
                        # Make processed labels
                        processed_points = []
                        
                        # Assumption: Processed images have equal height and width
                        if float(raw_image_width) / float(raw_image_height) > 1:
                            wider = True
                        else:
                            wider = False

                        # Adjust coordinates
                        for body_part_coordinates in raw_annotation:
                            body_part_x, body_part_y = tuple(map(float, body_part_coordinates[1:-1].split(',')))
                            if wider:
                                modified_body_part_x = body_part_x
                                modified_body_part_y = ((raw_image_height * body_part_y) + (float(raw_image_width - raw_image_height) / 2)) / raw_image_width
                            else:
                                modified_body_part_x = ((raw_image_width * body_part_x) + (float(raw_image_height - raw_image_width) / 2)) / raw_image_height
                                modified_body_part_y = body_part_y
                            processed_points.append((modified_body_part_x, modified_body_part_y))
                          
                    # Store processed image and labels
                    processed_image.save(os.path.join(dataset_processed_images_dir, dataset_image_name))
                    np.savetxt(os.path.join(dataset_processed_labels_dir, dataset_image_name[:dataset_image_name.rfind('.')] + '.txt'), processed_points, fmt='%.6f')
                    
def generate_resolution(resolution, project_constants):
    
    # Iterate over datasets
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
    
        # Make resolution directory
        dataset_processed_dir = os.path.join(project_constants.PROCESSED_DATA_DIR, dataset)
        default_processed_images_dir = os.path.join(dataset_processed_dir, 'images_{0}x{0}'.format(project_constants.MAXIMUM_RESOLUTION))
        resolution_processed_images_dir = os.path.join(dataset_processed_dir, 'images_{0}x{0}'.format(resolution))
        os.makedirs(resolution_processed_images_dir)
    
        # Resize images
        print('-- Resize {0} images'.format(dataset))
        for image_file in tqdm(os.listdir(default_processed_images_dir)):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                image = Image.open(os.path.join(default_processed_images_dir, image_file))
                resized_image = Image.fromarray(np.uint8(resize(np.array(image), resolution, resolution) * 255))
                resized_image.save(os.path.join(resolution_processed_images_dir, image_file))
                
        # Generate dataframe
        image_names = [(img_name[:img_name.rfind('.')], img_name[img_name.rfind('.')+1:]) for img_name in os.listdir(resolution_processed_images_dir) if img_name.endswith('.png') or img_name.endswith('.jpg')]
        image_ids = [img_id for img_id, img_type in image_names]
        image_paths = [os.path.join(resolution_processed_images_dir, img_id + '.' + img_type) for (img_id, img_type) in image_names]
        points_dir = os.path.join(dataset_processed_dir, 'points')
        points = [np.loadtxt(os.path.join(points_dir, img_id + '.txt')) for img_id in image_ids]
        df = pd.DataFrame(data={'id': image_ids, 'img_path': image_paths, 'points': points})
        df.set_index('id', inplace=True)
        df.to_hdf(os.path.join(dataset_processed_dir, 'data_{0}x{0}'.format(resolution)), dataset)
                
def process(project_dir, input_resolution, project_constants):
    print('\n============================================================================================================================================\n')
    print('PROCESSING DATA\n')
    
    # Process raw images and annotations
    processed_exists = os.path.exists(project_constants.PROCESSED_DATA_DIR)
    if not processed_exists:
        
        # Make processed directory
        os.makedirs(project_constants.PROCESSED_DATA_DIR)
        
        # Split images into datasets
        datasets_exists = os.path.exists(os.path.join(project_constants.RAW_IMAGES_DIR, 'train'))
        if not datasets_exists:
            print('\n- Generating datasets')
            generate_datasets(project_constants.RAW_IMAGES_DIR, project_constants.TRAINVAL_TEST_SPLIT, project_constants.TRAIN_VAL_SPLIT)
            print('- Datasets generated') 
        
        # Create processed images and corresponding labels
        print('\n- Processing images and labels')
        perform_processing(project_constants)
        print('- Images and labels processed')
        
    # Generate desired image resolution
    resolution_exists = os.path.exists(os.path.join(project_constants.PROCESSED_TRAIN_DIR, 'images_{0}x{0}'.format(input_resolution)))
    if not resolution_exists:
        print('\n- Enabling images of resolution {0}x{0}'.format(input_resolution)) 
        generate_resolution(input_resolution, project_constants)   
        print('- Resolution enabled')    
    print('\n============================================================================================================================================\n')
