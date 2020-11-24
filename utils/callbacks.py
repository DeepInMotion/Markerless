import csv
import os
import math
from tensorflow.keras.callbacks import Callback

import utils.evaluation as evaluation
import utils.datagenerator as datagenerator
        
class EvaluationHistory(Callback):
    
    def __init__(self, log_dir, eval_model, datagen, body_parts, head_segment, output_layer_index, flipped_body_parts, batch_size, confidence_threshold, preprocess_input=None, mpii=True, thresholds=[1.0, .5, .3, .1, .05], n_samples=None):
        self.datagen = datagen
        self.eval_model = eval_model
        self.preprocess_input = preprocess_input
        self.mpii = mpii
        self.body_parts = body_parts
        self.head_segment = head_segment
        self.output_layer_index = output_layer_index
        self.flipped_body_parts = flipped_body_parts 
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        self.thresholds = thresholds
        self.n_samples = n_samples
        
        os.makedirs(log_dir, exist_ok=True)
        file_names = [f'epochs_validation_pckh@{int(threshold * 100)}.csv' for threshold in self.thresholds]
        file_names.append('epochs_validation_error.csv')
        
        self.csv_paths = [os.path.join(log_dir, file_name) for file_name in file_names]
        
    def on_train_begin(self, logs=None):
        
        # Initialize CSV-files
        self.csv_files = []
        self.writers = []
        for csv_path in self.csv_paths:
            
            append_header = True
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    append_header = not bool(len(f.readline()))
                    
                csv_file = open(csv_path, 'a')
            else:
                csv_file = open(csv_path, 'w')
                    
            headers = ['epoch', 'mean'] + [body_part for body_part in self.body_parts]
            
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            if append_header:
                writer.writeheader() 
        
            self.csv_files.append(csv_file)
            self.writers.append(writer)
        
    def on_epoch_end(self, epoch, logs=None):
        
        # Evaluate
        results = self.eval_model.evaluate(self.datagen, preprocess_input=self.preprocess_input, thresholds=self.thresholds, n_samples=self.n_samples, mpii=self.mpii, head_segment=self.head_segment, body_parts=self.body_parts, output_layer_index=self.output_layer_index, flipped_body_parts=self.flipped_body_parts, batch_size=self.batch_size, confidence_threshold=self.confidence_threshold)
        
        # Write results to files
        print(f"\n---------------------------- Epoch: {epoch + 1}  ---------------------------\n")
        for i, key in enumerate(results):
            evaluation = results[key]
            
            if key == 'error':
                print(f'Mean keypoint error: {evaluation["mean"]}')
            else:
                threshold = key
                print(f'Mean keypoint accuracy@{int(threshold * 100)}: {evaluation["mean"]}')
                
            evaluation['epoch'] = epoch + 1

            self.writers[i].writerow(evaluation)
            self.csv_files[i].flush()
        print(f"\n-----------------------------------------------------------------\n")            
        
    def on_train_end(self, logs=None):
                      
        # Close files
        for csv_file in self.csv_files:
            csv_file.close()
        
        self.writers = None