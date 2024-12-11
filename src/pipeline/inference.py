import torch
import os
import cv2
import time
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from segment_anything import sam_model_registry, SamPredictor
from src.components.gpt import GPT
from src.components.yolo import Yolo
from src.components.ocr import OCR
from src.logging.logger import logging
from src.constants.count_approximation import estimate_tablet_count
from src.constants.generate_map import generate_map
from src.utils.image_handler import read_image, crop_image, enlarge_image, align_image

# Load environment variables from .env file
load_dotenv('.env')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("./artifacts/sam", "sam_vit_h_4b8939.pth")

print(f'Using: {DEVICE}')

# if this is able to support all the models at a time then we can proceed with this other wise we need to modularize this
class Inference():
    def __init__(self):
        self.sam = SamPredictor(sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE))
        self.gpt = GPT()
        self.yolo = Yolo()
        self.ocr = OCR()
        self.scale_factor = 0
        self.img_scale = 1.8
        self.pix_scale = 15
        self.area_real = 7.068 # this is cm^2
        self.strip_config = pd.read_csv('./assets/Tablet_Config.csv')
        self.count_threshold = 0.1 # this is the threshold for the count of the tablets, 10% of the total count.
        self.prompt = open('./assets/prompt.txt', 'r').read()
        # this is the client for the AWS Textract
        # self.client = boto3.client(os.getenv('AWS_SERVICE'),region_name=os.getenv('AWS_REGION'), aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'))

    def inference(self, image_path: str = None, image: Image = None) -> dict:
        
        t0 = time.time()

        result = {}

        # This function handles image reading from path and PIL image object
        image = read_image(image_path, image)
        
        # Getting the YOLO results, which are bounding boxes of the strips and a circle.
        yolo_res = self.yolo.inference(image = image)

        t1 = time.time()
        print(f'Time Taken for YOLO: {t1-t0:.2f}s')

        # This sets the image in the sam model, which will be used to generate maps, given approximate coordinates.
        self.sam.set_image(image)

        # Now we need to calculate the scale factor 
        mask, circle_area, box = generate_map(self.sam, image, yolo_res['circle'][0])

        # Cropping the circle
        cropped_image = crop_image(image, mask)
        # Now we need to calculate the scale factor
        self.scale_factor = self.area_real / circle_area
        t2 = time.time()

        print(f'Time Taken for Scale Factor Calculation along with SAM Inference: {t2-t1:.2f}s')
        
        for i, coords in enumerate(yolo_res['strip']):    
            # Getting the masked image

            mask, strip_pixel_area, box = generate_map(self.sam, image, coords)

            # Now we need to crop the image
            cropped_image = crop_image(image, mask)

            # Aligning the image
            aligned_image = align_image(cropped_image, mask)

            #cv2.imwrite(f'./runs/aligned_image{self.cnt}_{i + 1}.jpg', aligned_image)

            # Now we need to enlarge the image for better OCR results
            ocr_img = enlarge_image(aligned_image, self.img_scale)

            # cv2.imwrite(f'./runs/ocr_img{self.cnt}_{i + 1}.jpg', ocr_img)
            
            # Now we need to read the text from the image
            text = self.ocr.rotate_read(ocr_img)

            # -------- This block is for Text Detection using AWS Textract --------
            # _, buffer = cv2.imencode('.jpeg', cv2.cvtColor(ocr_img, cv2.COLOR_RGB2BGR))
            # ocr_img_bytes = bytearray(buffer)

            # tk = time.time()
            # response = self.client.detect_document_text(Document={'Bytes': ocr_img_bytes})
            # print(f'Time Taken for OCR of Strip {i + 1}: {time.time()-tk:.2f}s')
            # text = ""

            # for item in response["Blocks"]:
            #     if item["BlockType"] == "LINE" or item["BlockType"] == "WORD":
            #         text += item["Text"] + " "
            with open('results.txt', 'w') as f:
                f.write(text + '\n')
            # Now we need to append the prompt to the text
            if text != '':
                gpt_result = self.gpt.inference(text)

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
            if medicine_name in result:
                result[medicine_name]['Count'] += estimate_tablet_count(strip_estimated_area, self.curr_config['Area in cm2'], self.curr_config['Total Tablets'], self.count_threshold)
            else:
                result[medicine_name] = {'Count': estimate_tablet_count(strip_estimated_area, self.curr_config['Area in cm2'], self.curr_config['Total Tablets'], self.count_threshold), 'Area': strip_estimated_area}
            
            # print ('Medicine Name:', medicine_name)
            # print(f'Count: {result[medicine_name]["count"]} Area: {result[medicine_name]["Area"]:.2f}')

            result[medicine_name].update(gpt_result)
            result[medicine_name]['OCR'] = text
        # Noting the time taken for inference
        t3 = time.time()

        print(f'Time Taken for Strip Detection and OCR: {t3-t2:.2f}s')
        print(f'Time Taken for pipeline: {t3-t0:.2f}s')
        logging.info(f"Time taken for Pipeline: {t1-t0:.2f}s")
       # Returning the result (python dict)
        return result