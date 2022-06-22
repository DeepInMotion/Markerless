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
4. Create a new project folder under *'projects'* with a specified name (e.g., *'mpii2015'*) 
5. Create constants file (i.e., *'project_constants.py'*) within project folder to define keypoint setup etc.
6. Create a subfolder within your project folder with name *'data'* and within *'data'* two subfolders *'raw'* and *'processed'* (e.g., *'mpii2015/data/raw'* and *'mpii2015/data/processed'*).
7. Create a subfolder within your project folder with name *'experiments'*. Your result from training/evaluation will be stored in this folder 
8. Upload images and annotations
- Alternative a) If you have raw images not sorted into train, val, and test sets: Within the *'raw'* folder, upload your annotated images into an image folder named *'images'* (e.g., *'mpii2015/data/raw/images'*) and annotation file (i.e., *'annotations.csv'*) into *'annotations'* folder (e.g., *'mpii2015/data/raw/annotations'*). The procedure will randomize the images into *'train'*, *'val'*, and *'test'* folders and preprocess the images by resizing with zero-padding to images with height and width according to *MAXIMUM_RESOLUTION* (e.g., 1024x1024) in *'project_constants.py'*. 
- Alternative b) If you have preprocessed and sorted the images into train, val, and test: Directly upload the images into processed image folders (e.g., *'mpii2015/data/processed/train/images_1024x1024'*). In addition, upload annotations as txt files with identical file name as the images into a separate folder named *'points'* (e.g., *'mpii2015/data/processed/train/points'*).      
9. Set parameters of training and/or evaluation in main.py script
- Line 6:      Set name of your project folder
- Line 16:     Set name of the experiment. Your model and output data will be stored inside a folder with the experiment name
- Line 20-21:  Set True/False flag if you want to train (fine-tune) the networks and/or evaluate the network. Note that evaluate = True would refer to a existing                     experiment folder created when train = True 
- Line 23:     Set single or dual GPU use
- Line 35:     Set input resolution for images
- Line 39:     Set convnet type --> ['EfficientHourglass', 'EfficientPose', 'EfficientPose Lite', 'CIMA-Pose']
- Line 40-47:  If model_type = 'EfficientHourglass', set additional parameters
- Line 57-59:  Set training batch size, start epoch, and numbers of epochs in your training procedure
- Line 87-124: Training hyper-parameters, training schedule, data augmentation, etc can be set. However, the default parameters used are found to work very well for                  training of all convnets
10.  Save main.py (with chosen parameter setting)
11. Open new terminal window in jupyter lab
12. Type "python main.py" in terminal window and run training/evaluation of chosen convnet
13. Your result output will be stored in "experiments" folder
