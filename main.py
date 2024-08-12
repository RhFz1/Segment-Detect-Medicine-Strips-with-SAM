import torch
import os
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import warnings
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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

IMAGE_PATH = os.path.join(HOME, "images", "example_10.jpg")

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=10000)

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)
# sam_result = list(filter(lambda x : x['area'] > 16000 and x['area'] < 28000, sam_result))

# print(sam_result)
masks = [
    mask['segmentation']
    for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)
]
areas = [
    mask['area']
    for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)
]

for i, mask in enumerate(masks):
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(os.path.join(HOME, "example_maps", f"example_mask_{i+prev}.jpeg"))
    plt.show()

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

cv2.imwrite(os.path.join(HOME, "images", "example_annotated.jpeg"), annotated_image)