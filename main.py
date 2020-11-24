import os, sys


""" Project """

project_name = 'mpii2015'
project_dir = os.path.join('projects', project_name)
sys.path.append(project_dir)


""" Experiment details """

# Name of experiment
experiment_name = '24112020 2009 EfficientPose RT Lite'

# Flags
train = True #[True, False]
evaluate = True #[True, False]

# Model configuration
model_type = 'EfficientPose Lite' # ['EfficientHourglass', 'EfficientHourglass Lite', 'EfficientPose', 'EfficientPose Lite', 'CIMA-Pose']
input_resolution = 224
raw_output_resolution = {'EfficientHourglass': int(input_resolution / 4), 
                        'EfficientHourglass Lite': int(input_resolution / 4),
                        'EfficientPose': int(input_resolution / 8), 
                        'EfficientPose Lite': int(input_resolution / 8), 
                        'CIMA-Pose': int(input_resolution / 8)}[model_type] 
upscaled_output_resolution = input_resolution
training_output_layer = {'EfficientHourglass': 'stage1_confs_tune', #ENSURE CONSISTENCY WITH LAYER NAMING
                        'EfficientHourglass Lite': 'stage1_confs_tune', #ENSURE CONSISTENCY WITH LAYER NAMING
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
               'EfficientHourglass Lite': False,
               'EfficientPose': True, 
               'EfficientPose Lite': True, 
               'CIMA-Pose': False}[model_type]
output_type = {'EfficientHourglass': 'EH-1-TUNE', 
               'EfficientHourglass Lite': 'EH-1-TUNE',
               'EfficientPose': 'EP-1+2-PAFS-TUNE', 
               'EfficientPose Lite': 'EP-1+2-PAFS-TUNE', 
               'CIMA-Pose': 'CP-2-TUNE'}[model_type]

# Training hyperparameters
training_batch_size = 20
start_epoch = 0
num_epochs = 100
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

# Evaluation options
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
if model_type == 'EfficientHourglass': import models.efficienthourglass as m #ENSURE CONSISTENT NAMING OF SCRIPT
elif model_type == 'EfficientHourglass Lite': import models.efficienthourglass_lite as m #ENSURE CONSISTENT NAMING OF SCRIPT
elif model_type == 'EfficientPose': import models.efficientpose as m
elif model_type == 'EfficientPose Lite': import models.efficientpose_lite as m
elif model_type == 'CIMA-Pose': import models.cima_pose as m 
    
# Project constants
import project_constants as pc


""" GPU specifications  """

# Assign GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #["0", "1", ""]

# Specify GPU usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


""" Initialize experiment directories """
experiment_dir = os.path.join(project_dir, 'experiments', experiment_name)
weights_dir = os.path.join(experiment_dir, 'weights')
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)


""" Store experiment hyperparameters """

# Construct dictionary of hyperparameters
hyperparameters = {'model': {'model_type': model_type,
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

convnet = m.architecture(input_resolution=input_resolution, num_body_parts=pc.NUM_BODY_PARTS, num_segments=pc.NUM_SEGMENTS)
model = convnet.model
num_parameters, num_flops, num_ms, devices = summary.summary(model, upscaled_output_resolution=upscaled_output_resolution)


""" Training """

if train:
    
    # Initialize optimization process

    ## Initialize optimizer
    adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, decay=learning_rate_decay, amsgrad=amsgrad_flag)

    ## Initialize TensorBoard monitoring
    tensorboard_callbacks = TensorBoard(log_dir=experiment_dir, write_graph=True, batch_size=training_batch_size)

    ## Initialize checkpointing 
    checkpoint_path = weights_dir + '/weights.{epoch}.hdf5'
    checkpoint_callbacks = AltModelCheckpoint(checkpoint_path, convnet.model, save_best_only=False)

    ## Initialize evaluation during training
    raw_evaluation_model = evaluation.EvaluationModel(model=model, input_resolution=input_resolution, raw_output_resolution=raw_output_resolution)
    evaluation_callbacks = EvaluationHistory(log_dir=experiment_dir, datagen=val_datagenerator, eval_model=raw_evaluation_model, preprocess_input=m.preprocess_input, body_parts=pc.BODY_PARTS, mpii=False, thresholds=pckh_thresholds, head_segment=pc.HEAD_SEGMENT, output_layer_index=training_output_index, flipped_body_parts=pc.FLIPPED_BODY_PARTS, batch_size=training_batch_size, confidence_threshold=confidence_threshold)

    ## Compile model
    keras.losses.euclidean_loss = euclidean_loss
    model.compile(optimizer=adam, loss=[euclidean_loss for i in range(len(model.outputs))])

    # Perform training

    ## Training function
    def fit_model(model, train_data, val_data, train_generator, val_generator, epochs, initial_epoch=0):
        model.fit_generator(train_data,
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

        convnet.model.load_weights(os.path.join(weights_dir, 'weights.{0}.hdf5'.format(start_epoch)))

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
            epoch = int(row[0]) 
            mean_error = float(row[1])
            if mean_error < best_mean_error:
                best_epoch = epoch
                best_mean_error = mean_error
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