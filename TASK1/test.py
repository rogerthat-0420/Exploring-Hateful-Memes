import os
import cv2
import pandas as pd
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from collections import Counter
import jsonlines

# Specify the path to the folder containing images
img_folder = "./hateful_memes/"

# Get a list of 1000 image files from the folder
# image_files = sorted(os.listdir(img_folder))[:10]

jsonl_file_path = "./hateful_memes/train.jsonl"

# Initialize an empty list to store labels
labels = []
images_files = []

# Read the jsonl file and extract labels
with jsonlines.open(jsonl_file_path) as reader:
    for entry in reader:
        # labels.append(entry["label"])
        images_files.append(img_folder+entry['img'])


# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=["Image", "Detected_Classes", "Class_Frequencies"])

# Load the model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Initialize an empty list to store labels
labels = []
images_files = []

# Read the jsonl file and extract labels
with jsonlines.open(jsonl_file_path) as reader:
    for entry in reader:
        # labels.append(entry["label"])
        images_files.append(img_folder+entry['img'])


# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=["Image", "Detected_Classes", "Class_Frequencies"])
