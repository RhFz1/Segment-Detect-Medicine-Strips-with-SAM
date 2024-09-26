import torch
import os
import easyocr
import json
import time
import cv2
import json  
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from segment_anything import sam_model_registry, SamPredictor
from src.components.gpt import GPT
from src.components.yolo import StripCount
from src.logging.logger import logging
from src.constants.count_approximation import estimate_tablet_count
from src.constants.cv2_image_rotations import rotate_image
from src.constants.image_enhancements import enlarge_image
from src.constants.generate_map import generate_map
# Load environment variables from .env file
load_dotenv('.env')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("./artifacts/sam", "sam_vit_h_4b8939.pth")
# if this is able to support all the models at a time then we can proceed with this other wise we need to modularize this
class Inference():
    def __init__(self):
        self.sam = SamPredictor(sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE))
        self.gpt = GPT()
        self.yolo = StripCount()
        self.ocr = easyocr.Reader(['en'])
        self.scale_factor = 0
        self.pix_scale = 30
        self.area_real = 5.72555 # this is cm^2
        self.strip_config = pd.read_csv('./assets/Tablet_Config.csv')
        self.count_threshold = 0.1 # this is the threshold for the count of the tablets, 10% of the total count.
        self.prompt = open('./assets/prompt.txt', 'r').read()

    def inference(self, image_path: str = None, image: Image = None):

        self.result = {}
        # Check if the image_path and image is None
        if image_path is None and image is None:
            return {}

        t0 = time.time()
        # Perform inference using the sam model, which will return a list of dictionaries
        if image is not None:
            # Consider the case where the image is already loaded
            image = np.array(image)
        # Retaining the original image for cropping
        else:
            # Reading the image
            image = cv2.imread(image_path)
            # Converting the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        yolo_res = self.yolo.inference(image = image)


        self.sam.set_image(image)

        # Now we need to calculate the scale factor
        _, circle_area = generate_map(self.sam, image, yolo_res['circle'][0])

        self.scale_factor = self.area_real / circle_area

        print(f"Scale Factor: {self.scale_factor:.6f}, Area Real: {self.area_real}")
        
        for coords in yolo_res['strip']:    
            # Getting the masked image
            coords[:2] -= self.pix_scale
            coords[-2:] += self.pix_scale

            mask, strip_pixel_area = generate_map(self.sam, image, coords)

            # Find the bounding box of the mask (non-black region)
            y_indices, x_indices = np.where(mask)  # Get indices of the True values in the mask
            x_min, x_max = x_indices.min(), x_indices.max()  # Min and Max X coordinates
            y_min, y_max = y_indices.min(), y_indices.max()  # Min and Max Y coordinates

            # Crop the image to the bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]            
            # Now we need to enhance the image
            ocr_img = enlarge_image(cropped_image, 1.8)
            
            ocr_res = self.ocr.readtext(ocr_img, detail = 0, paragraph = True)
            ocr_rot_res = self.ocr.readtext(rotate_image(ocr_img, 270), detail = 0, paragraph = True)

            # Now we need to combine the results from the ocr
            text = ' '.join(ocr_res)
            text = text + ' '.join(ocr_rot_res)

            # Now we need to append the prompt to the text
            prompt = text + self.prompt

            # Now this text will be sent to GPT for results.
            gpt_result = self.gpt.inference(prompt)

            # converting the gpt_result to a dictionary
            gpt_result = json.loads(gpt_result)

            if gpt_result is None or 'Medicine_Name' not in gpt_result or gpt_result['Medicine_Name'] == '':
                continue
            
            # pull the medicine name from the gpt_result
            medicine_name = gpt_result['Medicine_Name']

            # Now we need to find the strip configuration for this medicine
            self.curr_config = self.strip_config.loc[self.strip_config['Tablet Name'] == medicine_name]

            if self.curr_config.empty:
                continue
            else:
                self.curr_config = self.curr_config.iloc[0].to_dict()

            strip_estimated_area = self.scale_factor * strip_pixel_area

            # Now lets count the tablets in the strip
            if medicine_name in self.result:
                self.result[medicine_name]['Count'] += estimate_tablet_count(strip_estimated_area, self.curr_config['Area in cm2'], self.curr_config['Total Tablets'], self.count_threshold)
            else:
                self.result[medicine_name] = {'Count': estimate_tablet_count(strip_estimated_area, self.curr_config['Area in cm2'], self.curr_config['Total Tablets'], self.count_threshold), 'Area': strip_estimated_area}
            
            # print ('Medicine Name:', medicine_name)
            # print(f'Count: {self.result[medicine_name]["count"]} Area: {self.result[medicine_name]["Area"]:.2f}')

            self.result[medicine_name].update(gpt_result)
        # Noting the time taken for inference
        t1 = time.time()
        
        logging.info(f"Time taken for Pipeline: {t1-t0:.2f}s")
       # Returning the result (python dict)
        return self.result