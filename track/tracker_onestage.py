# Dependencies
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm

from sys import platform
windows = False
if platform == "win32":
    windows = True

def track(video_dir, preprocessed_video_path):
    
    """ Experiment details """
    
    #### Step 1: Project name
    project_name = 'mpii2015' # <---Enter the name of your project folder
    
    project_dir = os.path.join('../projects', project_name)
    sys.path.append(project_dir)
    sys.path.append(os.path.join('..'))

    #### Step 2: Experiment name
    experiment_name = '30062022 1022 MPII2015_224x224_EfficientHourglassB0_Block1to6_weights' # <---Enter the name of your experiment

    #### Step 3: Decide if you want to perform predictions, save predicted coordinates as CSV file and/or visualize (i.e., annotate) predictions on video
    predict = True # <-- Assign [True, False] 
    save = True # <-- Assign  [True, False] 
    visualize = True # <-- Assign  [True, False] 
    fine_tune = True
    upscale = False

    #### Step 4: Choose model type and configuration. When using EfficientHourglass model, be aware of the comments and notes below. 
    model_type = 'EfficientHourglass' # <--assign model type ['EfficientHourglass', 'EfficientPose', 'EfficientPose Lite', 'CIMA-Pose']
    input_resolution = 224 # <-- assign resolution [Options for EfficientHourglass --> 128,160,192,224,256,288,320,356,384,(416),(448),(480),512, Options for EfficientPose --> 128,224,256,368,480,600, Options for EfficientPose Lite --> 128,224,256,368, Options for EfficientPose Lite --> 368]
    if model_type == 'EfficientHourglass':
        architecture_type = 'B' #<--assign architecture type for EfficientHourglass ['L'= EfficientHourglass_lite, 'B'= EfficientHourglass_original, 'H' = EfficientHourglass_lite_original_hybrid, 'X' = EfficientHourglass-X] Default is B
        efficientnet_variant = 0 #<--assign EfficientNet-backbone variant [Options --> 0, 1, 2, 3, 4] Default: 0
        block_variant = 'Block1to6' #<--assign number of blocks in the EfficientNet-backbone [Options --> 'Block1to5', (Block1to5b), 'Block1to6', 'Block1to7'] Default: Block1to6
        TF_version = None #<-- assign TF-version according to names og the weight files in 'pretrained' folder  [Options --> '_TF2', None]

    # Static hyperparameters (DO NOT CHANGE)
    ## Model configuration
    raw_output_resolution = {'EfficientHourglass': int(input_resolution / 4), 
                            'EfficientPose': int(input_resolution / 8), 
                            'EfficientPose Lite': int(input_resolution / 8), 
                            'CIMA-Pose': int(input_resolution / 8)}[model_type] 
    training_output_layer = {'EfficientHourglass': 'stage1_confs_tune', 
                            'EfficientPose': 'pass3_detection2_confs_tune', 
                            'EfficientPose Lite': 'pass3_detection2_confs_tune', 
                            'CIMA-Pose': 'stage2_confs_tune'}[model_type]
    training_output_index = {'EfficientHourglass': 0, 
                            'EfficientHourglass Lite': 0,
                            'EfficientPose': 2, 
                            'EfficientPose Lite': 2, 
                            'CIMA-Pose': 1}[model_type] 
    evaluation_output_index = None
    supply_pafs = {'EfficientHourglass': False, 
                   'EfficientPose': True, 
                   'EfficientPose Lite': True, 
                   'CIMA-Pose': False}[model_type]
    if fine_tune:
        output_type = {'EfficientHourglass': 'EH-1-TUNE', 
                       'EfficientPose': 'EP-1+2-PAFS-TUNE', 
                       'EfficientPose Lite': 'EP-1+2-PAFS-TUNE', 
                       'CIMA-Pose': 'CP-2-TUNE'}[model_type]
    else:
        output_type = {'EfficientHourglass': 'EH-1', 
                       'EfficientPose': 'EP-1+2-PAFS', 
                       'EfficientPose Lite': 'EP-1+2-PAFS', 
                       'CIMA-Pose': 'CP-2'}[model_type]

    if upscale:
        upscaled_output_resolution = input_resolution 
    else:
        upscaled_output_resolution = raw_output_resolution

    """ Dependencies """

    # External dependencies
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import backend as K
    import csv
    import numpy as np
    import skvideo.io
    import skvideo
    import math
    from PIL import Image, ImageDraw, ImageFont

    # Local dependencies
    import utils.summary as summary
    from utils.helpers import add_points, add_lines
    if model_type == 'EfficientHourglass':
        if fine_tune: import models.EfficientHourglass as m
        else: import models.EfficientHourglass_MPII as m
    elif model_type == 'EfficientPose': import models.efficientpose as m
    elif model_type == 'EfficientPose Lite': import models.efficientpose_lite as m
    elif model_type == 'CIMA-Pose': import models.cima_pose as m 

    # Project constants
    import project_constants as pc


    """ GPU specifications  """
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


    """ Define experiment directories """
    
    experiment_dir = os.path.join(project_dir, 'experiments', experiment_name)
    weights_dir = os.path.join(experiment_dir, 'weights')
    os.makedirs(os.path.join(experiment_dir, video_dir), exist_ok=True)

    """ Load model """
    
    if model_type == 'EfficientHourglass': 
        convnet = m.architecture(input_resolution=input_resolution, num_body_parts=pc.NUM_BODY_PARTS, num_segments=pc.NUM_SEGMENTS, architecture_type = architecture_type, efficientnet_variant = efficientnet_variant, block_variant = block_variant, TF_version = TF_version)
        convnet.model.save(os.path.join(experiment_dir, 'model.h5'))
        if(architecture_type == 'L' or architecture_type == 'H'):
            preprocess_input = m.preprocess_input_lite
        else:
            preprocess_input = m.preprocess_input
    else:
        convnet = m.architecture(input_resolution=input_resolution, num_body_parts=pc.NUM_BODY_PARTS, num_segments=pc.NUM_SEGMENTS)
        convnet.model.save(os.path.join(experiment_dir, 'model.h5'))
        preprocess_input = m.preprocess_input
    model = convnet.model
    raw_output = model.layers[-1].output
    upscaled_output = summary.upscale_block(raw_output, num_body_parts=pc.NUM_BODY_PARTS, raw_output_resolution=raw_output_resolution, upscaled_output_resolution=upscaled_output_resolution)
    upscaled_model = keras.Model(model.inputs, upscaled_output)

    # Load weights from epoch with smallest mean error
    best_epoch = None
    best_mean_error = 1.0
    with open(os.path.join(experiment_dir, 'epochs_validation_error.csv'), newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader)
        for row in reader:
            try:
                epoch = int(row[0])
                mean_error = float(row[1])
                if mean_error < best_mean_error:
                    best_epoch = epoch
                    best_mean_error = mean_error
            except:
                continue
    upscaled_model.load_weights(os.path.join(weights_dir, 'weights.{0}.hdf5'.format(best_epoch)), by_name=True)

    if predict:
        
        # General tracker constants
        body_parts_hpe = pc.BODY_PARTS
        num_body_parts_hpe = len(body_parts_hpe)
        batch_size = 128 # Batch size for HPE
        preprocess_batch_multi = 1 # DO NOT CHANGE!!! # multiplier of batch_size for preprocessing large batch for increase. Adapt to GPU to prevent OOM
        num_rotation = 0 # number of 90 degrees clock-wise rotations. Options 0 = 0 deg, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg 

        # Configurate GPU and TF sessions
        gpus = "0" #<--Assign "0" or "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        gpu_memory_fraction = 0.8
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        # CONFIGURATE HUMAN POSE ESTIMATION
        hpe_input_height, hpe_input_width = input_resolution, input_resolution
        hpe_output_height, hpe_output_width = upscaled_output_resolution, upscaled_output_resolution
        K.set_learning_phase(0)
        
        # Load video
        if not os.path.exists(preprocessed_video_path):
            print("Error: Tracker module could not find video")
        elif not preprocessed_video_path.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'wmv']:
            print("Error: Tracker module only supports MP4, AVI, MOV or WMV videos")
        else:
            try:
                videogen = skvideo.io.vreader(preprocessed_video_path)
                videogen_temp = skvideo.io.vreader(preprocessed_video_path)
                video_metadata = skvideo.io.ffprobe(preprocessed_video_path)['video']
                fps = skvideo.io.ffprobe(os.path.join(preprocessed_video_path))['video']['@r_frame_rate']
                num_video_frames = 0
                while True:
                    try:
                        next(videogen_temp)
                        num_video_frames += 1
                    except:
                        break   
                num_batches = int(np.ceil(num_video_frames / batch_size))
                frame_height, frame_width = next(skvideo.io.vreader(preprocessed_video_path)).shape[:2]
            except:
                print("Error: Tracker module could not load video")
        
        # Set batch and meta-batch numbers
        batch_num = int(np.ceil(num_video_frames/batch_size))
        meta_batch = int(np.ceil(batch_num/preprocess_batch_multi))
        
        coords = []
        metabatch_info = []
        print("Is GPU available for TensorFlow? {0}".format(tf.test.is_gpu_available()))
                
        for m in range(meta_batch):
            
            # Clear video_batch for new session
            video_batch = []
            video_batch2 = []
            batch_start = int(m*preprocess_batch_multi)
            batch_end = int((m+1)*preprocess_batch_multi)
                       
            for n in range(batch_start,batch_end):
                if (batch_end*batch_size) < num_video_frames:
                    batch = [next(videogen) for _ in range(batch_size)]
                else:
                    rest_frames = num_video_frames - (batch_start*batch_size)
                    batch = [next(videogen) for _ in range(rest_frames)]

                video_batch += list(batch)
            video_batch = np.array(video_batch, dtype = 'uint8')
            
            shape = np.shape(video_batch)
            max_shape = np.amax(shape[1:3])
            min_shape = np.amin(shape[1:3])
            min_arg = np.argmin([shape[2],shape[1]])

            # Set TF session for human pose estimation
            tf.compat.v1.keras.backend.clear_session()
            
            # Preprocess large video-batch in tensorflow
            video_batch = tf.constant(tf.convert_to_tensor(video_batch, dtype=tf.uint8))
            video_batch2 = tf.image.resize_with_pad(video_batch, target_height = hpe_input_height, target_width = hpe_input_width)
            if num_rotation > 0: video_batch2 = tf.image.rot90(video_batch2, k=num_rotation)
            video_batch2 = np.array(video_batch2)
            shape_hpe = np.shape(video_batch2)

            # Human pose estimation
            meta_batch_size = shape[0]
            batch_num = int(np.ceil(meta_batch_size/batch_size))
            
            beta = 9. 
            bdps = {i: 1.0 for i in range(num_body_parts_hpe)} # CAN BE ADJUSTED
            for nn in range(batch_num):
                start = nn*batch_size
                end = (nn+1)*batch_size 
                batch = video_batch2[start:end,:,:,:]
                mini_batch_size = batch.shape[0]

                # Preprocess images in batch
                batch = np.array(batch)
                batch = preprocess_input(batch)

                # Perform prediction
                batch_confs = upscaled_model.predict_on_batch(batch)

                # Extract coordinates
                batch_coords = []

                for n in range(mini_batch_size):
                    frame_coords = []
                    for i in range(num_body_parts_hpe):

                        # Find peak point
                        conf = batch_confs[n,...,i]
                        bdp_factor = bdps[i]
                        output_width = int(conf.shape[1])
                        output_height = int(conf.shape[0])
                        max_index = np.argmax(conf)
                        peak_y = float(math.floor(max_index / output_width))
                        peak_x = max_index % output_width
                        sigma = bdp_factor*(output_width/32) 

                        # Local soft-argmax
                        
                        # Define beta and size of local square neighborhood
                        num_pix = int(np.round(2*sigma))
                        num_pix1 = int(num_pix+1)
                        
                        # Define start and end indx for local square neighborhood
                        rows_start = int(np.max([int(peak_y)-num_pix, 0]))
                        rows_end = int(np.min([int(peak_y)+num_pix1, output_height]))
                        cols_start = int(np.max([int(peak_x)-num_pix, 0]))
                        cols_end = int(np.min([int(peak_x)+num_pix1, output_width]))
                        
                        # Define localsquare neigborhod 
                        loc_mat = conf[rows_start:rows_end,cols_start:cols_end]
                        y_ind = [i for i in range(rows_start,rows_end)]
                        x_ind = [j for j in range(cols_start,cols_end)]
                        posy, posx = np.meshgrid(y_ind, x_ind, indexing='ij')
                        
                        # Compute local soft-argmax for neigborhood
                        a = np.exp(beta*loc_mat)
                        b = np.sum(a)
                        softmax = a/b
                        peak_x = np.sum(softmax*posx)
                        peak_y = np.sum(softmax*posy)
                        peak_x += 0.5
                        peak_y += 0.5

                        # Normalize coordinates
                        peak_x /= conf.shape[1]
                        peak_y /= conf.shape[0]
                        
                        frame_coords.append([peak_x, peak_y])
                    batch_coords.append(frame_coords)

                # Convert to Numpy format
                batch_coords = np.asarray(batch_coords)
                coords += list(batch_coords)
                    
            # Redefine shape and dimensions after HPE preprocessing
            shape = np.shape(np.array(video_batch))
            num_images = shape[0]
            max_shape = np.amax(shape[1:3])
            min_shape = np.amin(shape[1:3])
            org_height = shape[1]
            org_width = shape[2]
            min_arg = np.argmin([shape[2],shape[1]])
            
            metabatch_info.append((num_images, min_shape, max_shape, org_height, org_width, min_arg))

        # Convert to original video coordinates
        coords=np.array(coords)
        org_coords=np.array(coords)
        coords_shape = np.shape(org_coords)
        adjust = 0
        for meta_batch, (num_images, min_shape, max_shape, org_height, org_width, min_arg) in enumerate(metabatch_info):
            if (num_rotation == 0 or num_rotation == 2):
                for i in range(0,num_images):
                    if (adjust+i)<coords_shape[0]:
                        norm_padding = (max_shape - min_shape) / (2 * max_shape)  
                        org_coords[adjust+i,:,min_arg] = (org_coords[adjust+i,:,min_arg] - norm_padding) / (1.0 - (2 * norm_padding))
            elif(num_rotation == 1 or num_rotation == 3):
                for i in range(0,num_images):
                    if (adjust+i)<coords_shape[0]:
                        norm_padding = (max_shape - min_shape) / (2 * min_shape) # VERIFY THIS ONE
                        org_coords[adjust+i,:,min_arg] = (org_coords[adjust+i,:,min_arg] - norm_padding) / (1.0 - (2 * norm_padding))
            adjust += num_images
        
        org_coords = np.asarray(org_coords[:num_video_frames])
    
    # Store coordinates to csv
    if save:
        body_parts_hpe = pc.BODY_PARTS
        headers = ['frame']
        [headers.extend([body_part + '_x', body_part + '_y']) for body_part in body_parts_hpe]
        if windows:
            video_name = preprocessed_video_path.split('\\')[-1].split('.')[0] 
        else:
            video_name = preprocessed_video_path.split('/')[-1].split('.')[0] #Mac: /, W10: \\
        
        # Initialize CSV
        try:
            os.mkdir(os.path.join(experiment_dir, video_dir, 'coords'))
        except OSError as error:
            pass 
        csv_path = os.path.join(experiment_dir, video_dir, 'coords', 'orgcoords_' + video_name + '.csv')
        csv_file = open(csv_path, 'w')
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()  

        # Write coordinates to CSV
        for j in range(org_coords.shape[0]):
            frame_coords = org_coords[j, ...]
            row = {'frame': j + 1}

            for body_part_coords, body_part in zip(frame_coords, body_parts_hpe):
                row[body_part + "_x"] = body_part_coords[0]
                row[body_part + "_y"] = body_part_coords[1]

            writer.writerow(row)

        csv_file.flush()
        csv_file.close()
        
    # Fetch predictions
    elif visualize:
        
        body_parts_hpe = pc.BODY_PARTS
        headers = ['frame']
        [headers.extend([body_part + '_x', body_part + '_y']) for body_part in body_parts_hpe]
        if windows:
            video_name = preprocessed_video_path.split('\\')[-1].split('.')[0] 
        else:
            video_name = preprocessed_video_path.split('/')[-1].split('.')[0] #Mac: /, W10: \\

        # Read CSV
        csv_path = os.path.join(experiment_dir, video_dir, 'coords', 'orgcoords_' + video_name + '.csv')
        csv_file = open(csv_path, 'r')
        reader = csv.DictReader(csv_file)
        org_coords = []
        for row in reader:
            frame_coords = []
            for body_part in body_parts_hpe:
                body_part_x = row[body_part + "_x"]
                body_part_y = row[body_part + "_y"]
                frame_coords.append([body_part_x, body_part_y])
            org_coords.append(frame_coords)
        org_coords = np.asarray(org_coords)
        
    # Visualize predictions
    if visualize:
    
        # Fetch video information
        video = skvideo.io.vreader(preprocessed_video_path)
        video_metadata = skvideo.io.ffprobe(preprocessed_video_path)['video']
        fps = video_metadata['@r_frame_rate']
        frame_height, frame_width = next(skvideo.io.vreader(preprocessed_video_path)).shape[:2]
        frame_side = frame_width if frame_width >= frame_height else frame_height

        # Initialize output video
        try:
            os.mkdir(os.path.join(experiment_dir, video_dir, 'annotations'))
        except OSError as error:
            pass 
        if windows:
            writer = skvideo.io.FFmpegWriter(os.path.join(experiment_dir, video_dir, 'annotations', 'tracked_' + video_name + '.mp4'), inputdict={'-r': fps}, outputdict={'-r': fps, '-vcodec': 'libx264', '-b': '30000000', '-pix_fmt': 'yuv420p'})
        else:
            writer = skvideo.io.FFmpegWriter(os.path.join(experiment_dir, video_dir, 'annotations', 'tracked_' + video_name + '.mp4'), inputdict={'-r': fps}, outputdict={'-r': fps, '-b': '30000000', '-pix_fmt': 'yuv420p'})
                                     
        # Annotate output video
        i = 0
        while True:
            try:
                frame = next(video)
                if num_rotation > 0: 
                    frame = np.array(frame, dtype = 'uint8')
                    frame = tf.constant(tf.convert_to_tensor(frame, dtype=tf.uint8))
                    frame = tf.image.rot90(frame, k=num_rotation)
                    frame = np.array(frame, dtype = 'uint8')
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                preds = org_coords[i]
                img = add_lines(img, preds, custom_height=frame_height, custom_width=frame_width, line_width=int(frame_side/200), associations=pc.SEGMENT_INDICES, colors=pc.BODY_PART_COLORS)
                img = add_points(img, preds, custom_height=frame_height, custom_width=frame_width, radius=int(frame_side/300), colors=pc.BODY_PART_COLORS)
                if windows:
                    font = ImageFont.truetype("arial.ttf", int(frame_width/40))
                else:
                    font = ImageFont.truetype("Keyboard.ttf", int(frame_width/40))
                draw.text((0.01*frame_width, 0.95*frame_height),"{0}".format(i+1),(255,255,255),font=font)
                writer.writeFrame(np.array(img))

                i += 1
            except:
                break

        writer.close()
        
    return org_coords
        
if __name__ == '__main__':
    # Fetch arguments
    args = sys.argv[1:]
    video_dir = args[0]
    files = os.listdir(video_dir)
    video_names = []
    for file in files:
        if '.' in file:
            file_format = file.split('.')[1].lower()
            if file_format in ['mp4', 'avi', 'mov', 'wmv']:
                video_names.append(file)
    for video_name in tqdm(video_names):
        track(video_dir, os.path.join(video_dir, video_name))