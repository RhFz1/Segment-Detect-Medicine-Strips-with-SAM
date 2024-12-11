import torch
import time
from PIL import Image
from dotenv import load_dotenv
from src.components.sam import SAM
from src.components.gpt import GPT
from src.components.yolo import Yolo
from src.components.ocr import OCR
from src.constants.count_approximation import GetCountApproximation
from src.logging.logger import logging
from src.utils.medicine_matching import MedicineMatcher
from src.utils.image_handler import read_image
from src.utils.common import abridge_results

# Load environment variables from .env file
load_dotenv('.env')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {DEVICE}')

# if this is able to support all the models at a time then we can proceed with this other wise we need to modularize this
class Inference():
    @torch.no_grad()
    def __init__(self):

        # Init all the models
        # SAM, GPT, YOLO, OCR
        self.sam = SAM()
        self.gpt = GPT()
        self.yolo = Yolo()
        self.ocr = OCR()
        self.medicine_matcher = MedicineMatcher()
        self.tablet_counter = GetCountApproximation()

    
    @torch.no_grad()
    def inference(self, image_path: str = None, image: Image = None, pred=False) -> dict:
        
        # Starting the inference
        # Setting time and result
        t0 = time.time()
        result = {}

        # This function handles image reading from path and PIL image object
        # If image is not provided, it reads the image from the path and vice versa.
        image = read_image(image_path, image)
        
        # Getting the YOLO results, which are bounding boxes of the strips and a circle.
        # This returns a dictionary with keys 'strip', 'circle' and 'box'
        yolo_res = self.yolo.inference(image = image)

        # Logging and printing the time taken for YOLO Inference
        t1 = time.time()
        logging.info(f"Time taken for YOLO Inference: {t1-t0:.2f}s")
        print(f"Time taken for YOLO Inference: {t1-t0:.2f}s")

        # This sets the image in the sam model.
        # And Also returns the scale factor for the image.
        self.sam.set_image(image)
        scale_factor = self.sam.get_scale_factor(yolo_res['circle'][0])
        
        # Logging the time taken for SAM Inference
        t2 = time.time()
        logging.info(f"Time taken for SAM Inference: {t2-t1:.2f}s")
        print(f"Time taken for SAM Inference: {t2-t1:.2f}s")
        
        # Starting the main pipeline
        # This handles the Segmentation, OCR and Counting of the tablets.
        # First let us estimate the number of strips
        k = len(yolo_res['strip'])
        for i, coords in enumerate(yolo_res['strip'] + yolo_res['box']):

            # Getting the cropped image and the pixel area of the strip
            # This will be used to estimate the count of the tablets in the strip.
            
            ocr_image, strip_pixel_area = self.sam.get_cropped_image(image, coords)

            # Logging the time taken for cropping the strip
            t3 = time.time()
            logging.info(f"Time taken for cropping strip-{i + 1}: {t3-t2:.2f}s")
            print(f"Time taken for cropping strip-{i + 1}: {t3-t2:.2f}s")
            
            # Perform OCR on the strip, which returns the text on the strip
            # Then using fuzzy logic to match the text with the medicine names
            # Skipping the strip if no medicine name is found
            text = self.ocr.perform_ocr(ocr_image)
            #print(text)
            medicine_name = self.medicine_matcher.get_name(text,pred)
            #print(medicine_name)
            if medicine_name is None:
                #gpt_result = self.gpt.inference(text,new_med=True)
                #print(gpt_result)
                #return gpt_result
                continue
            
            # Logging the time taken for OCR on the strip
            t4 = time.time()
            logging.info(f"Time taken for OCR on strip-{i + 1}: {t4-t3:.2f}s")
            print(f"Time taken for OCR on strip-{i + 1}: {t4-t3:.2f}s")

            # Perform GPT Inference on the text
            # This returns the details of the medicine
            gpt_result = self.gpt.inference(text)
            print(gpt_result)

            # Logging the time taken for GPT Inference
            t5 = time.time()
            logging.info(f"Time taken for GPT Inference on strip-{i + 1}: {t5-t4:.2f}s")
            print(f"Time taken for GPT Inference on strip-{i + 1}: {t5-t4:.2f}s")

            # Estimating the count of the tablets in the strip
            # Using the area calculated earlier
            # This returns a dictionary with keys 'Count' and 'Area'
            count_area = self.tablet_counter.get_count(medicine_name, strip_pixel_area * scale_factor, i <= (k - 1))
            #if count_area is None:
            #    continue

            # Logging the time taken for Counting the tablets
            t6 = time.time()
            logging.info(f"Time taken for Counting tablets in strip-{i + 1}: {t6-t5:.2f}s")
            print(f"Time taken for Counting tablets in strip-{i + 1}: {t6-t5:.2f}s")


            # Adding the medicine details to the result
            # This is a dictionary with the medicine name as the key and the details as the value
            result = abridge_results(medicine_name, gpt_result, count_area, result,i<=(k-1))
        # Logging the total time taken for the inference
        self.sam.unset_image()
        t7 = time.time()
        logging.info(f"Total time taken for Inference: {t7-t0:.2f}s")
        print(f"Total time taken for Inference: {t7-t0:.2f}s")
        torch.cuda.empty_cache()
        # Returning the result
        return result
    
    @torch.no_grad()
    def add_new_medicine(self, image_path: str = None, image: Image = None, pred=True) -> dict:
        """
        This function is used to add a new medicine to the inference results.
        It takes an image as input and returns the inference results.
        """
        result=self.inference(image_path=image_path,pred=pred)
        return result

    

if __name__ == '__main__':
    # Testing the inference
    inference = Inference()
    #or i in range(5):
    result = inference.inference(image_path = f'./assets/test_images/test.jpeg')
    print(result)
