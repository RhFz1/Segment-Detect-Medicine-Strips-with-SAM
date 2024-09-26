import torch
import time     
import os
import cv2
import warnings
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from src.logging.logger import logging
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("./artifacts/sam", "sam_vit_h_4b8939.pth")

class SAM():

    def __init__(self):
        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, stability_score_thresh=0.98)

    def inference(self, image_path: str = None, image = None):

        # Here im expecting an image path or a np.ndarray image
        # If image path is provided, load the image and convert it to RGB
        # If image is provided, use it directly after converting it to RGB

        t0 = time.time()
        # Load model
        if image is None:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_result = self.mask_generator.generate(image)

        t1 = time.time()

        # print(f"Time taken for SAM Map Generation: {t1-t0:.2f}s")

        logging.info(f"Time taken for SAM Map Generation: {t1-t0:.2f}s")
        
        return sam_result