import torch
import easyocr
import json
import time
import cv2
import json  
import numpy as np
import pandas as pd
from src.components.vit import ViT
from src.components.sam import SAM
from src.components.gpt import GPT
from src.logging.logger import logging
from src.constants.cv2_image_rotations import rotate_image

# if this is able to support all the models at a time then we can proceed with this other wise we need to modularize this
class Inference():
    def __init__(self):
        self.vit = ViT()
        self.sam = SAM()
        self.gpt = GPT()
        self.ocr = easyocr.Reader(['en'])
        self.num_maps = 4
        self.vit_threshold = 0.90
        self.dist_min = 1e9
        self.area_min = 1e9
        self.scale_factor = 0
        self.area_real = 5.72555 # this is cm^2
        self.strip_config = pd.read_csv('./assets/Tablet_Config.csv')
        self.result = {}

    def inference(self, image_path: str, image = None):

        t0 = time.time()
        # Perform inference using the sam model, which will return a list of dictionaries
        if image is not None:
            # Consider the case where the image is already loaded
            sam_result = self.sam.inference(image=image)
        else:
            sam_result = self.sam.inference(image_path=image_path) # this will return 
        
        # Retaining the original image for cropping
        if image is None:
            # Reading the image
            image = cv2.imread(image_path)
            # Converting the image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(min(self.num_maps, len(sam_result))):
            # Getting the mask area in pixels
            c_ar = sam_result[i]['area']
            # Getting the rough center of the mask
            cx, cy = sam_result[i]['point_coords'][0]
            # Getting the crop boundaries
            dy, dx = sam_result[i]['crop_box'][1:3]

            # getting the distance
            dist = np.sqrt((cx - dx)**2 + (cy - dy)**2)

            # check if the distance is less than the minimum distance
            if dist < self.dist_min:
                # swapping the area and distance
                self.area_min = c_ar
                self.dist_min = dist
        # calculate the scale factor
        self.scale_factor = self.area_real / self.area_min

        print(f"Scale Factor: {self.scale_factor}, Area Real: {self.area_real}, Area Min: {self.area_min} of reference.")
        
        # Perform some operations on the sam_result
        for i in range(min(self.num_maps, len(sam_result))):
            # Perform some operations on the sam_result
            mask = sam_result[i]['segmentation'].astype('uint8') * 255
            # Getting the bounding box
            x, y, w, h = sam_result[i]['bbox']            
            # pull out the mask
            cropped_mask = mask[y : y + h, x : x + w]
            # crop the image using the mask
            cropped_image = cv2.bitwise_and(image[y: y + h, x: x + w], image[y: y + h, x: x + w], mask=cropped_mask)

            # Get the strip area
            strip_pixel_area = sam_result[i]['area']

            # prediction
            prediction = self.vit.inference(image=cropped_image)
            
            # prediction is of this format tensor([[0.99]], device='cuda:0')
            if prediction.item() > self.vit_threshold:
                # Here we send the cropped image to ocr.
                ocr_res = self.ocr.readtext(cropped_image, detail = 0, paragraph = True)
                ocr_rot_res = self.ocr.readtext(rotate_image(cropped_image, 270), detail = 0, paragraph = True)

                text = ' '.join(ocr_res)
                text = text + ' '.join(ocr_rot_res)

                # Now this text will be sent to GPT for results.
                gpt_result = self.gpt.inference(text)

                # converting the gpt_result to a dictionary
                gpt_result = json.loads(gpt_result)
                
                # sometimes the gpt_result is a list, so we need to handle that
                if isinstance(gpt_result, list):
                    gpt_result = gpt_result[0]

                # pull the medicine name from the gpt_result
                medicine_name = gpt_result['Medicine_Name']

                # Now we need to find the strip configuration for this medicine
                self.curr_config = self.strip_config.loc[self.strip_config['Tablet Name'] == medicine_name]

                if self.curr_config.empty:
                    return {}
                else:
                    self.curr_config = self.curr_config.iloc[0].to_dict()

                # Now lets count the tablets in the strip
                if medicine_name in self.result:
                    self.result[medicine_name]['count'] += ((strip_pixel_area * self.scale_factor) / self.curr_config['Area in cm2']) * self.curr_config['Total Tablets']
                else:
                    self.result[medicine_name] = {'count': ((strip_pixel_area * self.scale_factor) / self.curr_config['Area in cm2']) * self.curr_config['Total Tablets'], 'Area': strip_pixel_area * self.scale_factor}
                print ('Medicine Name:', medicine_name)
                print(f'Count: {self.result[medicine_name]["count"]} Area: {self.result[medicine_name]["Area"]}')

                self.result[medicine_name].update(gpt_result)
        # Noting the time taken for inference
        t1 = time.time()
        # printing the time taken for inference
        # print(f"Time taken for Pipeline: {t1-t0:.2f}s")
        # logging the time taken for inference
        logging.info(f"Time taken for Pipeline: {t1-t0:.2f}s")
       # Returning the result (python dict)
        return self.result