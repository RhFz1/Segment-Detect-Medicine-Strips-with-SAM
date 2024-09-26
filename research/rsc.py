import torch
import os
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import cv2


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("./artifacts/sam", "sam_vit_h_4b8939.pth")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
predictor = SamPredictor(sam)


# Load the image
image = cv2.imread('./assets/img_006.jpg')

predictor.set_image(image)

# Define the coordinate box
input_box = np.array([761,  906, 1282, 1147])

# Perform prediction
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False
)

mask = masks[0]

# Calculate the area
area_in_pixels = np.sum(mask)
print(f"Segment area in pixels: {area_in_pixels}")

# Find the bounding box of the mask (non-black region)
y_indices, x_indices = np.where(mask)  # Get indices of the True values in the mask
x_min, x_max = x_indices.min(), x_indices.max()  # Min and Max X coordinates
y_min, y_max = y_indices.min(), y_indices.max()  # Min and Max Y coordinates

# Crop the image to the bounding box
cropped_image = image[y_min:y_max, x_min:x_max]

cv2.imwrite('cropped_segment.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))