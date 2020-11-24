import os


""" DIRECTORIES """

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')

# Raw data dir
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
RAW_IMAGES_DIR = os.path.join(RAW_DATA_DIR, 'images') 
RAW_ANNOTATIONS_DIR = os.path.join(RAW_DATA_DIR, 'annotations')
RAW_ANNOTATION_FILE = os.path.join(RAW_ANNOTATIONS_DIR, 'mpii_human_pose_v1_u12_1.mat') #os.path.join(RAW_ANNOTATIONS_DIR, 'annotations.csv')

# Processed data dir
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train')
PROCESSED_TRAIN_POINTS_DIR = os.path.join(PROCESSED_TRAIN_DIR, 'points') 
PROCESSED_VAL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val')
PROCESSED_VAL_POINTS_DIR = os.path.join(PROCESSED_VAL_DIR, 'points') 
PROCESSED_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test')
PROCESSED_TEST_POINTS_DIR = os.path.join(PROCESSED_TEST_DIR, 'points')

# Experiments dir
EXPERIMENTS_DIR = os.path.abspath('experiments')


""" DATASET """

# Data split
TRAINVAL_TEST_SPLIT = 0.80 # 80% trainval images, 20% test images
TRAIN_VAL_SPLIT = 0.91 # 91% train images, 9% val images

# Maximum image resolution
MAXIMUM_RESOLUTION = 1024

# Crop images
CROP = True # Define if images should be cropped
CROP_PADDING = 0.15 # 15% padding of each side of crop

# Body parts
BODY_PARTS = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
BODY_PART_COLORS = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
FLIPPED_BODY_PARTS = ['head_top', 'upper_neck', 'left_shoulder', 'left_elbow', 'left_wrist', 'thorax', 'right_shoulder', 'right_elbow', 'right_wrist', 'pelvis', 'left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle']
NUM_BODY_PARTS = len(BODY_PARTS)
SEGMENTS = [('head_top', 'upper_neck'), ('upper_neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('upper_neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('upper_neck', 'thorax'), ('thorax', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), ('thorax', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle')]
SEGMENT_INDICES = [(BODY_PARTS.index(body_part_a), BODY_PARTS.index(body_part_b)) for (body_part_a, body_part_b) in SEGMENTS] 
NUM_SEGMENTS = len(SEGMENTS)
HEAD_SEGMENT = (BODY_PARTS.index('head_top'), BODY_PARTS.index('upper_neck')) # Define upper and lower head keypoint used in computing PCKh