import torch
import time
import os
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import warnings
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from argparse import ArgumentParser

# Here I want to take in the prev variable as my argument
parser = ArgumentParser()
parser.add_argument("--prev", type=int, help="prev_count")
args = parser.parse_args()
prev = args.prev

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
HOME = '/home/ec2-user/FAIR/SAM'

IMAGE_PATH = os.path.join(HOME, "images", "example_4.jpeg")

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=10000)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

t0 = time.time()
sam_result = mask_generator.generate(image_rgb)
t1 = time.time()

print(f"Total time for Segmentation.\nTime: {t1 - t0:.3f}s")

for i in range(min(len(sam_result), 10)):
    # Assuming 'original_image' and 'sam_result' are available
    mask = sam_result[i]['segmentation'].astype(np.uint8) * 255
    x, y, w, h = sam_result[i]['bbox']
    cropped_mask = mask[y:y+h, x:x+w]
    cropped_image = cv2.bitwise_and(image_rgb[y:y+h, x:x+w], image_rgb[y:y+h, x:x+w], mask=cropped_mask)
    cv2.imwrite(os.path.join(HOME, "example_maps", f"example_cropped_{i + 1}.jpeg"), cropped_image)

t2=time.time()

print(f"Time: {t2 - t1:.3f}s")