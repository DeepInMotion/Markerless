import os, sys


""" Project """

project_name = 'mpii2015' #CHANGABLE
project_dir = os.path.join('projects', project_name)
sys.path.append(project_dir)


""" Experiment details """

# Options

## Name of experiment
experiment_name = '03032021 1709 MPII_128x128_EfficientHourglassB0_Block1to6_weights'

## Flags
train = True #[True, False] 
evaluate = True #[True, False] 

## GPU usage
Dual_GPU = False #True
if Dual_GPU:
    # Assign GPU
    gpus = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:  
    # Assign GPU
    gpus = "0" #"0", "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    gpu_memory_fraction = 0.8

## Model configuration 
input_resolution = 128 #[Options for EfficientHourglass --> 128,160,192,224,256,288,320,356,384,(416),(448),(480),512]
#input_resolution = [128, 160, 192, 224, 256, 288, 320, 356, 384, 512] #Vector for batch processing
upscaled_output_resolution = input_resolution 

model_type = 'EfficientHourglass' # ['EfficientHourglass', 'EfficientPose', 'EfficientPose Lite', 'CIMA-Pose']
if model_type == 'EfficientHourglass':
    #Set architecture parameters for EfficientHourglass
    architecture_type = 'B' #['L'= EfficientHourglass_lite, 'B'= EfficientHourglass_original, 'H' = EfficientHourglass_lite_original_hybrid, 'X' = EfficientHourglass-X]
    # Default: B
    efficientnet_variant = 0 #[Options --> 0, 1, 2, 3, 4] Default: 0
    block_variant = 'Block1to6' #[Options --> 'Block1to5', (Block1to5b), 'Block1to6', 'Block1to7'] Default: Block1to6
    # NOTE: If error running 'Block1to5', change file name in 'pretrained' to 'Block1to5b' and then change block_variant = 'Block1to5b'
    GPU = None #[Options --> '_3090', 'None']

#NOTE1: The 'pretrained' folder shows all possible combinations of architecture parameters (Total of 141 options).
#NOTE2: When using 'EfficientHourglass', please make sure that architecture parameters is consistent with the file name in the 'pretrained' folder.
# Example: 
# File name --> MPII_224x224_EfficientHourglassB0_Block1to6_weights --> 
# MPII_{input_resolution}x{input_resolution}_EfficientHourglass{architecture_type}{efficientnet_variant}_{block_variant}_weights{GPU} -->
# input_resolution = 224, architecture_type = 'B', efficientnet_ variant = 0, block_variant = 'Block1to6', GPU = None

## Training hyperparameters
training_batch_size = 16 
start_epoch = 0 
num_epochs = 50 

# Static hyperparameters

## Model configuration
raw_output_resolution = {'EfficientHourglass': int(input_resolution / 4), 
                        'EfficientPose': int(input_resolution / 8), 
                        'EfficientPose Lite': int(input_resolution / 8), 
                        'CIMA-Pose': int(input_resolution / 8)}[model_type] 
training_output_layer = {'EfficientHourglass': 'stage1_confs_tune', #ENSURE CONSISTENCY WITH LAYER NAMING, DONE
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
output_type = {'EfficientHourglass': 'EH-1-TUNE', 
               'EfficientPose': 'EP-1+2-PAFS-TUNE', 
               'EfficientPose Lite': 'EP-1+2-PAFS-TUNE', 
               'CIMA-Pose': 'CP-2-TUNE'}[model_type]

## Train hyperparameters
if model_type == 'EfficientHourglass':
    # Vector description --> (sigma, None, epoch, None)
    # NOTE: If comparing models across different resolution, make sure that all sigma values are scaled to the raw_output_resolution
    schedule = {32: [(3.5, None, 0, None),(3, None, 2, None),(2.5, None, 4, None),(2, None, 6, None),(1.75, None, 8, None),(1.625, None, 12, None),(1.5, None, 16, None),(1.375, None, 20, None),(1.25, None, 25, None),(1.125, None, 30, None),(1, None, 35, None),(1, None, 40, None)],
                40: [(4.375, None, 0, None),(3.75, None, 2, None),(3.125, None, 4, None),(2.5, None, 6, None),(2.1875, None, 8, None),(2.03125, None, 12, None),(1.875, None, 16, None),(1.71875, None, 20, None),(1.5625, None, 25, None),(1.40625, None, 30, None),(1.25, None, 35, None),(1.09375, None, 40, None)],
                48: [(5.25, None, 0, None),(4.5, None, 2, None),(3.75, None, 4, None),(3, None, 6, None),(2.625, None, 8, None),(2.4375, None, 12, None),(2.25, None, 16, None),(2.0625, None, 20, None),(1.875, None, 25, None),(1.6875, None, 30, None),(1.5, None, 35, None),(1.3125, None, 40, None)],
                56: [(6.125, None, 0, None),(5.25, None, 2, None),(4.375, None, 4, None),(3.5, None, 6, None),(3.0625, None, 8, None),(2.84375, None, 12, None),(2.625, None, 16, None),(2.40625, None, 20, None),(2.1875, None, 25, None),(1.96875, None, 30, None),(1.75, None, 35, None),(1.53125, None, 40, None)],
                64: [(7.0, None, 0, None),(6.0, None, 2, None),(5.0, None, 4, None),(4.0, None, 6, None),(3.5, None, 8, None),(3.25, None, 12, None),(3.0, None, 16, None),(2.75, None, 20, None),(2.5, None, 25, None),(2.25, None, 30, None),(2, None, 35, None),(1.75, None, 40, None)],
                72: [(7.875, None, 0, None),(6.75, None, 2, None),(5.625, None, 4, None),(4.5, None, 6, None),(3.9375, None, 8, None),(3.65625, None, 12, None),(3.375, None, 16, None),(3.09375, None, 20, None),(2.8125, None, 25, None),(2.53125, None, 30, None),(2.25, None, 35, None),(1.96875, None, 40, None)],
                80: [(8.75, None, 0, None),(7.5, None, 2, None),(6.25, None, 4, None),(5, None, 6, None),(4.375, None, 8, None),(4.0625, None, 12, None),(3.75, None, 16, None),(3.4375, None, 20, None),(3.125, None, 25, None),(2.8125, None, 30, None),(2.5, None, 35, None),(2.1875, None, 40, None)],
                88: [(9.625, None, 0, None),(8.25, None, 2, None),(6.875, None, 4, None),(5.5, None, 6, None),(4.8125, None, 8, None),(4.46875, None, 12, None),(4.125, None, 16, None),(3.78125, None, 20, None),(3.4375, None, 25, None),(3.09375, None, 30, None),(2.75, None, 35, None),(2.40625, None, 40, None)],
                96: [(10.5, None, 0, None),(9.0, None, 2, None),(7.5, None, 4, None),(6, None, 6, None),(5.25, None, 8, None),(4.875, None, 12, None),(4.5, None, 16, None),(4.125, None, 20, None),(3.75, None, 25, None),(3.375, None, 30, None),(3, None, 35, None),(2.625, None, 40, None)],
                104: [(11.375, None, 0, None),(9.75, None, 2, None),(8.125, None, 4, None),(6.5, None, 6, None),(5.6875, None, 8, None),(5.28125, None, 12, None),(4.875, None, 16, None),(4.46875, None, 20, None),(4.0625, None, 25, None),(3.65625, None, 30, None),(3.25, None, 35, None),(2.84375, None, 40, None)],
                112: [(12.25, None, 0, None),(10.5, None, 2, None),(8.75, None, 4, None),(7, None, 6, None),(6.125, None, 8, None),(5.6875, None, 12, None),(5.25, None, 16, None),(4.8125, None, 20, None),(4.375, None, 25, None),(3.9375, None, 30, None),(3.5, None, 35, None),(3.0625, None, 40, None)],
                120: [(13.125, None, 0, None),(11.25, None, 2, None),(9.375, None, 4, None),(7.5, None, 6, None),(6.5625, None, 8, None),(6.09375, None, 12, None),(5.625, None, 16, None),(5.15625, None, 20, None),(4.6875, None, 25, None),(4.21875, None, 30, None),(3.75, None, 35, None),(3.28125, None, 40, None)],
                128: [(14, None, 0, None),(12, None, 2, None),(10, None, 4, None),(8, None, 6, None),(7, None, 8, None),(6.5, None, 12, None),(6, None, 16, None),(5.5, None, 20, None),(5, None, 25, None),(4.5, None, 30, None),(4, None, 35, None),(3.5, None, 40, None)]}[raw_output_resolution]
else:
    schedule = {16: [(1.8, 0.87, 0, None), (1.64, 0.79, 2, None), (1.48, 0.71, 6, None),(1.32, 0.625, 14, None),(1.25, 0.563, 22, None),(1.163, 0.547, 30, None),(1.075, 0.532, 38, None), (0.988, 0.516, 46, None), (0.9, 0.5, 54, None)], 
            28: [(3.1, 1.53, 0, None),(2.6, 1.31, 2, None),(2.2, 1.09, 6, None),(1.75, 0.875, 14, None),(1.53, 0.788, 22, None),(1.422, 0.766, 30, None),(1.313, 0.744, 38, None),(1.203, 0.722, 46, None),(1.1, 0.7, 54, None)],
            32: [(3.5, 1.75, 0, None), (3.0, 1.5, 2, None), (2.5, 1.25, 6, None),(2.0, 1.0, 14, None),(1.75, 0.9, 22, None),(1.625, 0.875, 30, None),(1.5, 0.85, 38, None), (1.375, 0.825, 46, None), (1.25, 0.8, 54, None)],
            46: [(5.0, 2.5, 0, None), (4.3, 2.15, 2, None), (3.6, 1.8, 6, None),(2.875, 1.4, 14, None),(2.5, 1.3, 22, None),(2.336, 1.258, 30, None),(2.156, 1.222, 38, None), (1.977, 1.186, 46, None), (1.797, 1.15, 54, None)],
            60: [(6.6, 3.3, 0, None), (5.6, 2.81, 2, None), (4.7, 2.34, 6, None), (3.75, 1.875, 14, None), (3.28, 1.688, 22, None), (3.05, 1.641, 30, None), (2.813, 1.594, 38, None), (2.578, 1.547, 46, None), (2.344, 1.5, 54, None)],
            75: [(8.2, 4.1, 0, None), (7.0, 3.51, 2, None), (5.9, 2.92, 6, None), (4.69, 2.343, 14, None), (4.10, 2.109, 22, None), (3.81, 2.050, 30, None), (3.516, 1.992, 38, None), (3.223, 1.934, 46, None), (2.930, 1.875, 54, None)]}[raw_output_resolution]
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
learning_rate_decay = 0.0
amsgrad_flag = True
augmentation_rotation = 45
augmentation_zoom = 0.25
augmentation_flip = True

## Evaluation options
evaluation_batch_size = 49
pckh_thresholds = [3.0, 2.0, 1.0, .5, .3, .1, .05]
confidence_threshold = 0.0001
flip = False


""" Dependencies """

# External dependencies
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from alt_model_checkpoint.tensorflow import AltModelCheckpoint
import csv
import json

# Local dependencies
import utils.process as process
import utils.datagenerator as datagenerator
import utils.summary as summary
import utils.evaluation as evaluation
from utils.callbacks import EvaluationHistory
from utils.losses import euclidean_loss
if model_type == 'EfficientHourglass': import models.EfficientHourglass as m #ENSURE CONSISTENT NAMING OF SCRIPT, DONE
elif model_type == 'EfficientPose': import models.efficientpose as m
elif model_type == 'EfficientPose Lite': import models.efficientpose_lite as m
elif model_type == 'CIMA-Pose': import models.cima_pose as m 
    
# Project constants
import project_constants as pc


""" GPU specifications  """

# Specify GPU usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


""" Initialize experiment directories """
experiment_dir = os.path.join(project_dir, 'experiments', experiment_name)
weights_dir = os.path.join(experiment_dir, 'weights')
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)


""" Store experiment hyperparameters """

# Construct dictionary of hyperparameters
hyperparameters = {'gpu_usage': {'gpus': gpus,
                                 'gpu_memory_fraction': gpu_memory_fraction},
                   'model': {'model_type': model_type,
                            'input_resolution': input_resolution,
                            'raw_output_resolution': raw_output_resolution,
                            'upscaled_output_resolution': upscaled_output_resolution,
                            'training_output_layer': training_output_layer,
                            'training_output_index': training_output_index,
                            'evaluation_output_index': evaluation_output_index,
                            'supply_pafs': supply_pafs,
                            'output_type': output_type},
                  'training': {'training_batch_size': training_batch_size,
                              'start_epoch': start_epoch,
                              'num_epochs': num_epochs,
                              'schedule': schedule,
                              'learning_rate': learning_rate,
                              'beta1': beta1,
                              'beta2': beta2,
                              'learning_rate_decay': learning_rate_decay,
                              'amsgrad_flag': amsgrad_flag,
                              'augmentation_rotation': augmentation_rotation,
                              'augmentation_zoom': augmentation_zoom,
                              'augmentation_flip': augmentation_flip},
                  'evaluation': {'evaluation_batch_size': evaluation_batch_size,
                                'pck_thresholds': pckh_thresholds,
                                'conficence_threshold': confidence_threshold,
                                'flip': flip}}

# Store hyperparameters as JSON file
with open(os.path.join(experiment_dir, 'hyperparameters.json'), 'w') as json_file:  
    json.dump(hyperparameters, json_file)


""" Initialize data """

# Process images and annotations based on desired resolutions (assuming raw folder exists with images and annotation file)
process.process(project_dir, input_resolution, project_constants=pc)

# Initialize datagenerators
train_df = pd.read_hdf(os.path.join(pc.PROCESSED_TRAIN_DIR, 'data_{0}x{0}'.format(str(input_resolution))), 'train')
val_df = pd.read_hdf(os.path.join(pc.PROCESSED_VAL_DIR, 'data_{0}x{0}'.format(str(input_resolution))), 'val')
datagenerator_settings = {'input_size': (input_resolution, input_resolution), 
                     'output_size': (raw_output_resolution, raw_output_resolution),
                     'batch_size': training_batch_size, 
                     'aug_rotation': augmentation_rotation, 
                     'aug_zoom': augmentation_zoom, 
                     'aug_flip': augmentation_flip, 
                     'body_parts': pc.BODY_PARTS, 
                     'flipped_body_parts': pc.FLIPPED_BODY_PARTS}
train_datagenerator = datagenerator.DataGenerator(df=train_df, settings=datagenerator_settings)
val_datagenerator = datagenerator.DataGenerator(df=val_df, settings=datagenerator_settings)
  
    
""" Initialize model """

if model_type == 'EfficientHourglass': 
    convnet = m.architecture(input_resolution=input_resolution, num_body_parts=pc.NUM_BODY_PARTS, num_segments=pc.NUM_SEGMENTS, architecture_type = architecture_type, efficientnet_variant = efficientnet_variant, block_variant = block_variant, GPU = GPU)
    convnet.model.save(os.path.join(experiment_dir, 'model.h5'))
else:
    convnet = m.architecture(input_resolution=input_resolution, num_body_parts=pc.NUM_BODY_PARTS, num_segments=pc.NUM_SEGMENTS)

if Dual_GPU:
    with tf.device('/cpu:0'):
        model0 = convnet.model
    model = multi_gpu_model(model0, gpus=2)
else:
    model = convnet.model
num_parameters, num_flops, num_ms, devices = summary.summary(model, upscaled_output_resolution=upscaled_output_resolution)


""" Training """

if train:
    
    # Initialize optimization process

    ## Initialize optimizer
    adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=learning_rate_decay, amsgrad=amsgrad_flag)

    ## Initialize TensorBoard monitoring
    tensorboard_callbacks = TensorBoard(log_dir=experiment_dir, write_graph=True)

    ## Initialize checkpointing 
    checkpoint_path = weights_dir + '/weights.{epoch}.hdf5'
    checkpoint_callbacks = AltModelCheckpoint(checkpoint_path, convnet.model, save_best_only=False) #Check TF format

    ## Initialize evaluation during training
    raw_evaluation_model = evaluation.EvaluationModel(model=model, input_resolution=input_resolution, raw_output_resolution=raw_output_resolution)
    evaluation_callbacks = EvaluationHistory(log_dir=experiment_dir, datagen=val_datagenerator, eval_model=raw_evaluation_model, preprocess_input=m.preprocess_input, body_parts=pc.BODY_PARTS, mpii=False, thresholds=pckh_thresholds, head_segment=pc.HEAD_SEGMENT, output_layer_index=training_output_index, flipped_body_parts=pc.FLIPPED_BODY_PARTS, batch_size=training_batch_size, confidence_threshold=confidence_threshold)

    ## Compile model
    keras.losses.euclidean_loss = euclidean_loss
    model.compile(optimizer=adam, loss=[euclidean_loss for i in range(len(model.outputs))])

    # Perform training

    ## Training function
    def fit_model(model, train_data, val_data, train_generator, val_generator, epochs, initial_epoch=0):
        model.fit(train_data,
                                      steps_per_epoch=train_generator.n_steps(),
                                      epochs=epochs,
                                      validation_data=val_data,
                                      validation_steps=val_generator.n_steps(),
                                      callbacks=[tensorboard_callbacks, checkpoint_callbacks, evaluation_callbacks],
                                      initial_epoch=initial_epoch, 
                                      workers=0)

    ## Initialize training (fine-tuning)
    if start_epoch == 0: 
        train_data = train_datagenerator.get_data(batch_size=training_batch_size, schedule=schedule, shuffle=True, augment=True, supply_pafs=supply_pafs, model_type=output_type, preprocess_input=m.preprocess_input, segments=pc.SEGMENT_INDICES) 
        val_data = val_datagenerator.get_data(batch_size=training_batch_size, schedule=schedule, shuffle=False, augment=False, supply_pafs=supply_pafs, model_type=output_type, preprocess_input=m.preprocess_input, segments=pc.SEGMENT_INDICES) 

        convnet.model.load_weights(convnet.pretrained_path, by_name=True)

        fit_model(model, train_data, val_data, train_datagenerator, val_datagenerator, num_epochs) 

    ## Continue training from last epoch
    else:
        train_data = train_datagenerator.get_data(batch_size=training_batch_size, schedule=schedule, initial_epoch=start_epoch, shuffle=True, augment=True, supply_pafs=supply_pafs, model_type=output_type, preprocess_input=m.preprocess_input, segments=pc.SEGMENT_INDICES)
        val_data = val_datagenerator.get_data(batch_size=training_batch_size, schedule=schedule, initial_epoch=start_epoch, shuffle=False, augment=False, supply_pafs=supply_pafs, model_type=output_type, preprocess_input=m.preprocess_input, segments=pc.SEGMENT_INDICES)

        convnet.model.load_weights(os.path.join(weights_dir, 'weights.{0}.hdf5'.format(start_epoch))) #Check TF format

        fit_model(model, train_data, val_data, train_datagenerator, val_datagenerator, num_epochs, initial_epoch=start_epoch)

        
""" Evaluation """

if evaluate:
          
    # Load correct model with upscaling
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
    
    # Evaluate model precision
    upscaled_evaluation_model = evaluation.EvaluationModel(model=upscaled_model, input_resolution=input_resolution, raw_output_resolution=raw_output_resolution)
    validation_results = upscaled_evaluation_model.evaluate(val_datagenerator, preprocess_input=m.preprocess_input, thresholds=pckh_thresholds, mpii=False, flip=flip, head_segment=pc.HEAD_SEGMENT, body_parts=pc.BODY_PARTS, output_layer_index=evaluation_output_index, flipped_body_parts=pc.FLIPPED_BODY_PARTS, batch_size=evaluation_batch_size, confidence_threshold=confidence_threshold)

    # Evaluate model efficiency
    validation_results['num_parameters'] = num_parameters
    if num_flops:
        validation_results['num_flops'] = num_flops
    if num_ms:
        validation_results['num_ms'] = num_ms 
        validation_results['fps'] = 1/(num_ms/1000) 
    validation_results['devices'] = devices
        
    # Store validation results as JSON file
    with open(os.path.join(experiment_dir, 'validation_results.json'), 'w') as json_file:  
        json.dump(validation_results, json_file)
