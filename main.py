import torch
import time
import os
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import warnings
import shutil
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from argparse import ArgumentParser
from inference import predict

# Here I want to take in the prev variable as my argument
# parser = ArgumentParser()
# parser.add_argument("--prev", type=int, help="prev_count")
# args = parser.parse_args()
# prev = args.prev

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
HOME = '/home/ec2-user/FAIR/SAM'

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=10000)


nval = os.listdir(os.path.join(HOME, "m2_train_images", 'train', 'No'))[-1].split('.')[0].split('_')[-1]
yval = os.listdir(os.path.join(HOME, "m2_train_images", 'train', 'Yes'))[-1].split('.')[0].split('_')[-1]

prev = max(int(nval), int(yval))
num_maps = 24

for i, image_path in enumerate(os.listdir(os.path.join(HOME, "images"))):
    t0 = time.time()
    image_path = os.path.join(HOME, "images", image_path)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t1 = time.time()
    sam_result = mask_generator.generate(image_rgb)
    t2 = time.time()
    print(f"Total time for Segmentation of {i + 1}th image. Time: {t2 - t1:.4f}s")

    for j in range(min(len(sam_result), num_maps)):
        # Assuming 'original_image' and 'sam_result' are available
        mask = sam_result[j]['segmentation'].astype(np.uint8) * 255
        x, y, w, h = sam_result[j]['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped_mask = mask[y:y+h, x:x+w]
        cropped_image = cv2.bitwise_and(image_rgb[y:y+h, x:x+w], image_rgb[y:y+h, x:x+w], mask=cropped_mask)
        cv2.imwrite(os.path.join(HOME, "example_maps", f"example_cropped_{prev + j + 1}.jpeg"), cropped_image)
    prev += num_maps

    for j, img in enumerate(os.listdir(os.path.join(HOME, "example_maps"))):

        img_path = os.path.join(HOME, "example_maps", img)
        yes_img_dir = os.path.join(HOME, "m2_train_images", 'train', 'Yes')
        no_img_dir = os.path.join(HOME, "m2_train_images", 'train', 'No')

        if predict(img_path).item() > 0.3:
            shutil.move(img_path, yes_img_dir)
        else:
            shutil.move(img_path, no_img_dir)

    t3 =time.time()
    
    print(f"{i + 1}th Image processed, Time: {t3 - t0: .4f}")