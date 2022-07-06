# Markerless

## How to setup on a Windows machine

### GPU activation

We strongly advise the use of workstation with NVIDIA GPU to speed up training of models. To enable use of GPU, follow these instructions:
1. Download [Visual Studio 2017 Free Community Edition](https://www.techspot.com/downloads/downloadnow/6278/?evp=ec1cdb914a1b435daaf013a4a084b093&file=7630) and install the program by following the necessary steps.
2. Download [CUDA Toolkit 11.1 Update 1](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and follow instructions to perform installation.
3. Copy the file *'ptxas.exe'* in the folder *'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\'* to *'Desktop'*.
4. Download [CUDA Toolkit 11.0 Update 1](https://developer.nvidia.com/cuda-11.0-update1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and follow instructions to perform installation.
5. Copy the file *'ptxas.exe'* from *'Desktop'* to the folder *'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\'*.
6. Create a user at [NVIDIA.com](https://developer.nvidia.com/login) and download [CUDNN 8.0.4](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.0_20200923/cudnn-11.0-windows-x64-v8.0.4.30.zip).
7. Open *'cudnn-11.0-windows-x64-v8.0.4.30.zip'* in *'Downloads'* and move the files in the folders *'bin'*, *'include'*, and *'lib'* under *'cuda'* to associated folders (*'bin'*, *'include'*, and *'lib'*) in *'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\'*.
8. Restart the computer.

### Setup Markerless framework

To setup the Markerless framework, follow these instructions:
1. Download [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) and perform the installation.
2. Open a command prompt and clone the Markerless framework: **git clone https://github.com/DeepInMotion/Markerless.git**
3. Navigate to the Markerless folder: **cd Markerless**
4. Create the virtual environment markerless: **conda env create -f environment.yml**

## How to use on a Windows machine

This is a step by step description for how to use the Markerless framework:
1. Open a command prompt and activate the virtual environment: **activate markerless**
2. Navigate to the Markerless folder: **cd Markerless**
3. Open the code library in a web browser: **jupyter lab**
4. Create a new project folder under *'projects'* with a specified name (e.g., *'mpii2015'*).
5. Create constants file (i.e., *'project_constants.py'*) within project folder to define keypoint setup etc.
6. Create a subfolder within your project folder with name *'experiments'* (e.g., *'mpii2015/experiments'*). Your results from training and evaluation will be stored in this folder.
7. Create a subfolder within your project folder with name *'data'* (e.g., *'mpii2015/data'*).
8. Upload images and annotations:
- Alternative a) If you have raw images not sorted into train, val, and test sets: Create a subfolder *'raw'* within *'data'*, and upload your annotated images into an image folder named *'images'* (e.g., *'mpii2015/data/raw/images'*) and annotation file (i.e., *'annotations.csv'*) into *'annotations'* folder (e.g., *'mpii2015/data/raw/annotations'*). The procedure will randomize the images into *'train'*, *'val'*, and *'test'* folders and preprocess the images by resizing with zero-padding to images with height and width according to `MAXIMUM_RESOLUTION` (e.g., 1024x1024) in *'project_constants.py'*. 
- Alternative b) If you have preprocessed and sorted the images into train, val, and test: Create a subfolder *'processed'* within the *'data'* folder and directly upload the images into separate dataset image folders (e.g., *'mpii2015/data/processed/train/images_1024x1024'*). In addition, for each dataset upload annotations as txt files with identical file name as the images into a separate folder named *'points'* (e.g., *'mpii2015/data/processed/train/points'*).      
9. Set parameters of training and/or evaluation in *'main.py'*:
- Line 8: Set name of your project folder.
- Line 19: Set name of the experiment. Your model and output data will be stored inside a folder with the given experiment name within the *'experiments'* subfolder.
- Line 22: Set `train = True` if you want to train the ConvNet, otherwise set `train = False` to skip training. 
- Line 23: If `train = True`, set `fine_tune = True` if you want to fine-tune the ConvNet, otherwise use `fine_tune = False` to perform training from scratch. 
- Line 24: Set `evaluate = True` if you want to evaluate the ConvNet, otherwise use `evaluate = False`. The evaluation will be performed on the model placed in the folder given by the experiment name. 
- Line 28: Set `Dual_GPU = True` for dual GPU use, otherwise `Dual_GPU = False` for single GPU.
- Line 40: Set ConvNet type, either EfficientHourglass, EfficientPose, EfficientPose Lite, or CIMA-Pose (e.g., `model_type = 'EfficientHourglass'`).
- Line 41: Set input resolution of images (e.g., `input_resolution = 224`).
- Line 43-46: If `model_type = 'EfficientHourglass'`, set additional parameters.
- Line 56-58: Set training batch size (e.g., `training_batch_size = 16 `), start epoch of training (e.g., `start_epoch = 0`), and numbers of epochs in a training run (e.g., `num_epochs = 50`).
- Line 61-70: Hyperparameters for training optimization, data augmentation etc. can be set. However, the default parameters are found to work very well for                  training of all the included ConvNets.
- Line 73-76: Set preferences for the evaluation process, including batch size (e.g., `evaluation_batch_size = 16`), PCK<sub>h</sub> thresholds to evaluate (e.g., `pckh_thresholds = [3.0, 2.0, 1.0, .5, .3, .1, .05]`), confidence threshold for a prediction to be performed (e.g., `confidence_threshold = 0.0001`), and flip evaluation (i.e., `flip = True` for combining predictions of original and flipped images, otherwise `flip = False`).
10. Save *'main.py'* (with the chosen hyperparameter setting).
11. Open a new terminal window from the jupyter lab tab in the web browser.
12. Run training and/or evaluation of the chosen ConvNet in the terminal window: **python main.py**
13. The results of the training and evaluation processes will be stored in the folder of the current experiment within the *'experiments'* folder (e.g., *'mpii2015/experiments/30062022 1022 MPII2015_224x224_EfficientHourglassB0_Block1to6_weights'*).

**Tip: The batch script (i.e., *'main_batch.py'*) may be used for sequential training of the same ConvNet with different input resolutions to determine the optimal model complexity.**
