# Markerless

## How to setup on a Windows machine

### GPU activation

We strongly advise the use of workstation with NVIDIA GPU to speed up training of models. To enable use of GPU, follow these instructions:
1. Download [Visual Studio 2017 Free Community Edition](https://www.techspot.com/downloads/downloadnow/6278/?evp=ec1cdb914a1b435daaf013a4a084b093&file=7630) and install the program by following the necessary steps.
2. Download [CUDA Toolkit 11.1 Update 1](https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and follow instructions to perform installation.
3. Copy the file **'ptxas.exe'** in the folder **'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin\'** to **'Desktop'**.
4. Download [CUDA Toolkit 11.0 Update 1](https://developer.nvidia.com/cuda-11.0-update1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal) and follow instructions to perform installation.
5. Copy the file **'ptxas.exe'** from **'Desktop'** to the folder **'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\'**.
6. Create a user at [NVIDIA.com](https://developer.nvidia.com/login) and download [CUDNN 8.0.4](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.4/11.0_20200923/cudnn-11.0-windows-x64-v8.0.4.30.zip).
7. Open **'cudnn-11.0-windows-x64-v8.0.4.30.zip'** in **'Downloads'** and move the files in the folders **'bin'**, **'include'**, and **'lib'** under **'cuda'** to associated folders (**'bin'**, **'include'**, and **'lib'**) in **'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\'**.
8. Restart the computer.

### Setup Markerless framework

To setup the Markerless framework, follow these instructions:
1. Download [Anaconda](https://docs.anaconda.com/anaconda/install/windows/) and perform the installation.
2. Open a command prompt and clone the Markerless framework: **git clone https://github.com/DeepInMotion/Markerless.git**
3. Navigate to the Markerless folder: **cd Markerless**
4. Create the virtual environment **tf2**: **conda env create -f environment.yml**

## How to use on a Windows machine

This is a step by step description for how to use the Markerless framework:
1. Open a command prompt and activate the virtual environment: **activate tf2**
2. Navigate to the Markerless folder: **cd Markerless**
3. Open the code library in a web browser: **jupyter lab**
4. Create a new project folder under "projects" with a specified name (e.g. "mpii2015") 
5. Create two subfolders within your project folder with name "data" and a "raw" and "processed" sub-folders (e.g. "mpii2015/data/raw" and "mpii2015/data/processed")
6. Upload your annotated images in a image folder named "??" within the "raw" folder (e.g. "mpii2015/data/raw/images"). The prosedure will randomise the images into a        "train/val/test"-folder and preprocess the images by resample them with zero-padding to images with 1024x1024 resolution (or resolution in line ??)
7. If you have preprocessed and sorted the images, they can be uploaded into e.g. "images_1024x1024" within a train/val/test folder (e.g. mpii2015/data/processed/train, mpii2015/data/processed/val, and mpii2015/data/processed/test)    
...
