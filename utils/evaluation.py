import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cdist
from tensorflow import keras
from keras.models import load_model
import keras.losses
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageDraw

from utils.helpers import extract_point, resize, show_confidence_maps
from utils.losses import euclidean_loss

keras.losses.euclidean_loss = euclidean_loss

def evaluate_one(gt, pred, head_segment_indices, threshold=0.5, min_head_segment=.001, max_head_segment=1.0, default_head_segment=.1, mpii=True, human_errors=None):
    """ Calculate PCKh evaluation given an annotation and a set of predicted points
    
    Parameters
    ----------
    gt: [(x0, y0), (x1, y1), ...]
        List of ground truth points
    pred: [(x0, y0), (x1, y1), ...]
        List of predicted points
    threshold: float
        A threshold controlling the propotion of the head segment which a point can be off,
        and still be concidered a correct prediction.
        
    Returns
    ----------
    correct_points: [Bool]
        A list of booleans indicating whether the point was correctly predicted or not
    """
    
    # Fetch indices of head segment
    if threshold is not None:
        head_indices = head_segment_indices
        head_top_index = head_indices[0]
        upper_neck_index = head_indices[1]

        # Account for missing body parts (consider efficiency of this block)
        missing_head_segment = False
        for i in range(gt.shape[0]):
            bp_coords = gt[i, ...]
            if bp_coords[0] == 0.0 and bp_coords[1] == 0.0:
                if i == head_top_index or i == upper_neck_index:
                    missing_head_segment = True
                pred[i, ...] = np.asarray([0.0, 0.0]) # Assumption: Missing body parts considered as correct predictions

        # Compute length of head segment    
        if missing_head_segment:
            head_segment = default_head_segment
        else:
            if mpii:
                head_segment = 0.75 * euclidean(gt[head_top_index], gt[upper_neck_index]) # Approximate head size with 0.75, is really 0.6 * diagonal of head rectangle
            else:
                head_segment = euclidean(gt[head_top_index], gt[upper_neck_index]) #0.6 * euclidean(gt[head_top_index], gt[upper_neck_index]) # Don't know why they scale the head size!!
            head_segment = np.clip(head_segment, min_head_segment, max_head_segment)  
    
    # Compute error of predictions
    distances = cdist(gt, pred, metric='euclidean').diagonal()

    # Compute accuracy
    if threshold is not None:
        correct_points = distances <= head_segment * threshold
    else:
        correct_points = distances <= human_errors
    
    return correct_points

def evaluate_all(Y_gt, Y_pred, head_segment, threshold=0.5, mpii=True, human_errors=None):
    """ Calculate mean accuracy for each point in the predicted Y_pred, given ground truth annotations in Y_gt.
    
    Parameters
    ----------
    Y_gt: ndarray of shape (n_samples, n_points, 2)
        Nd-array of ground truth annotations for each sample
    Y_pred: ndarray of shape (n_samples, n_points, 2)
        Nd-array of predicted points for each sample
    threshold: float
        A threshold controlling the propotion of the head segment which a point can be off,
        and still be concidered a correct prediction.
        
    Returns
    ----------
    accuracy: float
        The mean accuracy of all predictions, over all points
    accuracy_per_point: [float] of shape (n_points, )
        The mean accuracy for each point
    """
    correct_predictions = [evaluate_one(gt, pred, threshold=threshold, head_segment_indices=head_segment, mpii=mpii, human_errors=human_errors) for gt, pred in zip(Y_gt, Y_pred)]
    accuracy_per_point = np.mean(correct_predictions, axis=0)
    accuracy = np.mean(accuracy_per_point)

    return accuracy, accuracy_per_point

def prediction_error(Y_gt, Y_pred):
    
    distances = []
    for y_gt, y_pred in zip(Y_gt, Y_pred):
        
        bodypart_distances = []
        for p_gt, p_pred in zip(y_gt, y_pred):
            dist = np.sqrt((p_gt[0] - p_pred[0])**2 + (p_gt[1] - p_pred[1])**2)
            bodypart_distances.append(dist)
        
        distances.append(bodypart_distances)
    
    mean_pr_point = np.mean(distances, axis=0)
    
    return mean_pr_point

class EvaluationModel():
   
    def __init__(self, input_resolution, raw_output_resolution, model_path=None, model=None):
        if model==None:
            self.model = load_model(model_path)
        else:
            self.model = model
        self.input_resolution = input_resolution
        self.output_size = (raw_output_resolution, raw_output_resolution)
        self.scale_factor = self.input_resolution / self.output_size[0]
    
    def predict_on_batch(self, img_batch, body_parts, output_index, flipped_body_parts, confidence_threshold, ids_batch=None, scales=[1.0], flip=False, output_scale=1.0, smooth=False, smooth_stride=2, augmented_val_dir=None, augmented_preprocess_input=False, smooth_type='average', confidence=False): 
        all_scales_confs = []
        for scale in scales:
            if scale < 1.0:
                padding = int(int((self.output_size[0] - (scale * self.output_size[0])) / 2) * self.scale_factor)
                temp_img_batch = np.asarray([resize(np.pad(img, ((padding,padding), (padding,padding), (0,0)), mode='constant', constant_values=0), self.input_resolution, self.input_resolution) for img in img_batch])
            elif scale > 1.0:
                cut = int(int(((scale * self.output_size[0]) - self.output_size[0]) / 2) * self.scale_factor)
                if cut > 0:
                    temp_img_batch = np.asarray([resize(img[cut:-cut, cut:-cut, :], self.input_resolution, self.input_resolution) for img in img_batch])
                else:
                    temp_img_batch = np.asarray([resize(img, self.input_resolution, self.input_resolution) for img in img_batch])
            elif smooth: # Smoothing only active for default scale and does not currently support flipping
                augmented_img_batch = []
                images_stride = {}
                i = 0
                for image_id in ids_batch:
                    image_info = image_id.split('_')
                    image_name = image_info[0]
                    frame_number = int(image_info[1])
                    relevant_frames = [n for n in range(frame_number-smooth_stride, frame_number+smooth_stride+1)]
                    highest_stride = 0
                    for stride in range(1, smooth_stride+1):
                        lower_image_path = os.path.join(augmented_val_dir, image_name + '_{0}.jpg'.format(frame_number-stride))
                        upper_image_path = os.path.join(augmented_val_dir, image_name + '_{0}.jpg'.format(frame_number+stride))
                        if os.path.exists(lower_image_path) and os.path.exists(upper_image_path):
                            lower_image = img_to_array(load_img(lower_image_path))
                            upper_image = img_to_array(load_img(upper_image_path))
                            if augmented_preprocess_input:
                                lower_image = augmented_preprocess_input(np.expand_dims(lower_image, axis=0))[0,...]
                                upper_image = augmented_preprocess_input(np.expand_dims(upper_image, axis=0))[0,...]
                            augmented_img_batch.append(lower_image)
                            augmented_img_batch.append(upper_image)
                            highest_stride = stride
                        else:
                            break
                    augmented_img_batch.append(img_batch[i])
                    images_stride[image_id] = highest_stride
                    i += 1
                    
                temp_img_batch = np.asarray(augmented_img_batch)    
                
            else:
                temp_img_batch = img_batch
        
            preds = self.model.predict_on_batch(temp_img_batch)
            if output_index is not None and not output_index == 0:
                scale_confs = preds[output_index]
            else:
                scale_confs = preds
                
            if flip and not smooth:
                flipped_preds = self.model.predict_on_batch(np.asarray([np.fliplr(img) for img in temp_img_batch]))
                if output_index is not None and not output_index == 0:
                    flipped_scale_confs = flipped_preds[output_index]
                else:
                    flipped_scale_confs = flipped_preds
                final_confs = []
                for image_confs, image_flipped_confs in zip(scale_confs, flipped_scale_confs):
                    unflipped_sorted_image_confs = [None for i in range(len(body_parts))]
                    for n in range(len(body_parts)):
                        flipped_body_part = flipped_body_parts[n]
                        corresponding_index = body_parts.index(flipped_body_part)
                        unflipped_sorted_image_confs[corresponding_index] = np.fliplr(image_flipped_confs[...,n])
                    unflipped_sorted_image_confs = np.moveaxis(np.asarray(unflipped_sorted_image_confs), 0, -1)
                    final_image_confs = (image_confs + unflipped_sorted_image_confs) * 0.5
                    final_confs.append(final_image_confs)
                
                scale_confs = np.asarray(final_confs)
            if scale < 1.0:
                if padding > 0:
                    output_padding = int(padding/self.scale_factor)
                    scale_confs = np.asarray([resize(conf, self.output_size[0]+2*output_padding, self.output_size[1]+2*output_padding) for conf in scale_confs])[:, output_padding:-output_padding, output_padding:-output_padding, :]
                else:
                    scale_confs = np.asarray([resize(conf, self.output_size[0], self.output_size[1]) for conf in scale_confs]) 
            elif scale > 1.0:
                output_cut = int(cut/self.scale_factor)
                scale_confs = np.asarray([np.pad(resize(conf, self.output_size[0]-2*output_cut, self.output_size[1]-2*output_cut), ((output_cut,output_cut), (output_cut,output_cut), (0,0)), mode='constant', constant_values=0) for conf in scale_confs])
            all_scales_confs.append(scale_confs)
        num_scales = len(scales)
        confs = []
        for image in range(scale_confs.shape[0]):
            image_confs = all_scales_confs[0][image, ...]
            for s in range(1, num_scales):
                image_confs += all_scales_confs[s][image, ...]
            confs.append(np.asarray(image_confs * (1 / num_scales)))
        if smooth: # Compute average prediction based on window around image
            raw_points = [self.predict_on_sample(conf, confidence_threshold=confidence_threshold, body_parts=body_parts) for conf in confs]
            current_index = 0
            points = []
            for image_id in ids_batch:
                image_stride = images_stride[image_id]
                relevant_points = raw_points[current_index:current_index+(2*image_stride)+1]
                window_size = len(relevant_points)
                frame_points = []
                for i in range(len(body_parts)):
                    if smooth_type == 'average':
                        total_x = 0.0
                        total_y = 0.0
                        for current_points in relevant_points:
                            x, y = current_points[i]
                            total_x += x
                            total_y += y
                        frame_points.append((total_x / window_size, total_y / window_size))
                    elif smooth_type == 'median':
                        xs = []
                        ys = []
                        for current_points in relevant_points:
                            x, y = current_points[i]
                            xs.append(x)
                            ys.append(y)
                        xs = sorted(xs)
                        ys = sorted(ys)
                        frame_points.append((xs[image_stride], ys[image_stride]))
                
                points.append(frame_points)
                current_index += window_size
            
        else:
            points = [self.predict_on_sample(conf, confidence_threshold=confidence_threshold, body_parts=body_parts, confidence=confidence) for conf in confs]
            
        return np.asarray(points)
    
    def inference_on_batch(self, img_batch, body_parts, output_index, flipped_body_parts, flip=False): 
            
        # Perform inference
        preds = self.model.predict_on_batch(img_batch)
        
        # Extract confidence maps
        if output_index is not None and not output_index == 0:
            raw_confs = preds[output_index]
        else:
            raw_confs = preds
                
        # In case of flipped images
        if flip:
            confs = []
            for image_confs in raw_confs:
                unflipped_sorted_image_confs = [None for i in range(len(body_parts))]
                for n in range(len(body_parts)):
                    flipped_body_part = flipped_body_parts[n]
                    corresponding_index = body_parts.index(flipped_body_part)
                    unflipped_sorted_image_confs[corresponding_index] = np.fliplr(image_confs[...,n])
                unflipped_sorted_image_confs = np.moveaxis(np.asarray(unflipped_sorted_image_confs), 0, -1)
                confs.append(unflipped_sorted_image_confs)
            confs = np.asarray(confs)
        else:
            confs = raw_confs
                        
        return confs
        
    def predict_on_sample(self, confs, body_parts, confidence_threshold, confs_size=None, raw_image_information=None, image=None, confidence=False):
        
        points = []

        for bp in body_parts:
            idx = body_parts.index(bp)
            point = extract_point(confs[..., idx], threshold=confidence_threshold, confidence=confidence)
            points.append(point)

        return np.asarray(points)

    def evaluate(self, val_datagen, head_segment, body_parts, output_layer_index, flipped_body_parts, batch_size, confidence_threshold, preprocess_input=None, n_samples=None, supply_ids=False, thresholds=[0.5], display_pred=False, display_gt=False, mpii=True, flip=False, smooth=False, smooth_stride=2, augmented_val_dir=None, augmented_preprocess_input=False, smooth_type='average', standard_deviation=False, shuffle=False, human_errors=None):
        
        accs = []
        
        accuracies = {threshold: [] for threshold in thresholds}
        if human_errors is not None:
            accuracies['Human'] = []
        errors = []
        
        datagen = val_datagen.get_eval_data(preprocess_input=preprocess_input, n_samples=n_samples, supply_ids=supply_ids, batch_size=batch_size, shuffle=shuffle)
        if supply_ids:
            for ids_batch, img_batch, point_batch in datagen:
                    pred_point_batch = self.predict_on_batch(img_batch, ids_batch=ids_batch, body_parts=body_parts, output_index=output_layer_index, flip=flip, flipped_body_parts=flipped_body_parts, smooth=smooth, smooth_stride=smooth_stride, augmented_val_dir=augmented_val_dir, augmented_preprocess_input=augmented_preprocess_input, smooth_type=smooth_type, confidence_threshold=confidence_threshold)

                    for threshold in thresholds:
                        acc, acc_pr_point = evaluate_all(point_batch, pred_point_batch, head_segment, threshold=threshold, mpii=mpii)
                        accuracies[threshold].append(acc_pr_point)
                    if human_errors is not None:
                        acc, acc_pr_point = evaluate_all(point_batch, pred_point_batch, head_segment, threshold=None, mpii=mpii, human_errors=human_errors)
                        accuracies['Human'].append(acc_pr_point)

                    errors.append(prediction_error(point_batch, pred_point_batch))

                    if display_pred:
                        for i, (img, points) in enumerate(zip(img_batch, pred_point_batch)):
                            plt.imshow(img)

                            for px, py in points:
                                px *= img.shape[1]
                                py *= img.shape[0]

                                plt.plot(px, py, 'o')
                            plt.show()

                    if display_gt:
                        img1 = Image.fromarray(np.uint8(img_batch[0,...]*255))
                        points1 = list(point_batch[0])
                        img1_draw = ImageDraw.Draw(img1)
                        for x, y in points1:
                            x *= self.input_resolution
                            y *= self.input_resolution
                            img1_draw.ellipse([x-5, y-5, x+5, y+5], fill='blue')
                        del img1_draw

                        plt.imshow(img1)
                        plt.show()
        else:
            for img_batch, point_batch in datagen:
                pred_point_batch = self.predict_on_batch(img_batch, body_parts=body_parts, output_index=output_layer_index, flip=flip, flipped_body_parts=flipped_body_parts, confidence_threshold=confidence_threshold)

                for threshold in thresholds:
                    acc, acc_pr_point = evaluate_all(point_batch, pred_point_batch, head_segment, threshold=threshold, mpii=mpii)
                    accuracies[threshold].append(acc_pr_point)
                if human_errors is not None:
                    acc, acc_pr_point = evaluate_all(point_batch, pred_point_batch, head_segment, threshold=None, mpii=mpii, human_errors=human_errors)
                    accuracies['Human'].append(acc_pr_point)

                errors.append(prediction_error(point_batch, pred_point_batch))

                if display_pred:
                    for i, (img, points) in enumerate(zip(img_batch, pred_point_batch)):
                        plt.imshow(img)

                        for px, py in points:
                            px *= img.shape[1]
                            py *= img.shape[0]

                            plt.plot(px, py, 'o')
                        plt.show()

                if display_gt:
                    img1 = Image.fromarray(np.uint8(img_batch[0,...]*255))
                    points1 = list(point_batch[0])
                    img1_draw = ImageDraw.Draw(img1)
                    for x, y in points1:
                        x *= self.input_resolution
                        y *= self.input_resolution
                        img1_draw.ellipse([x-5, y-5, x+5, y+5], fill='blue')
                    del img1_draw

                    plt.imshow(img1)
                    plt.show()
                    
        results = {}
        
        # Accuracy
        for threshold in thresholds:
            res = {}
            
            accs = accuracies[threshold]
            acc_pr_point = np.mean(accs, axis=0)
            mean_acc = np.mean(acc_pr_point)
            res['mean'] = mean_acc

            for i, bp in enumerate(body_parts):
                res[bp] = acc_pr_point[i]
                
            results[threshold] = res
        if human_errors is not None:
            res = {}
            
            accs = accuracies['Human']
            acc_pr_point = np.mean(accs, axis=0)
            mean_acc = np.mean(acc_pr_point)
            res['mean'] = mean_acc

            for i, bp in enumerate(body_parts):
                res[bp] = acc_pr_point[i]
                
            results['Human'] = res
        
        # Error
        res = {}
        err_pr_point = np.mean(errors, axis=0)
        stdev_pr_point = np.std(errors, axis=0)
        err_mean = np.mean(err_pr_point)
        res['mean'] = err_mean
        
        for i, bp in enumerate(body_parts):
            if standard_deviation:
                res[bp] = (err_pr_point[i], stdev_pr_point[i])
            else:
                res[bp] = err_pr_point[i]
        
        if standard_deviation:
            results['error (mean, standard deviation)'] = res
        else:
            results['error'] = res
        
        return results