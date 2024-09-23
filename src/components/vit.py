import torch
import time     
import os
import cv2
import torch.nn as nn
import warnings
from PIL import Image
from src.components.preprocess import data_transforms
from src.components.vit_arch import Model 
from src.logging.logger import logging

warnings.filterwarnings("ignore")

model_path = './artifacts/vit/vit.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ViT():
    def __init__(self):
        # Initialize the model
        self.model = Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model = self.model.to(device)

    def inference(self, image_path:str = None, image: cv2 = None):

        t0 = time.time()
        # Load the image
        if image is None:
            image = Image.open(image_path).convert('RGB')
            
        # Apply transformations

        image = data_transforms(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Move the image to the appropriate device    
        image = image.to(device)

        # Perform inference
        with torch.no_grad():
            output = self.model(image)
            prediction = output
        
        t1 = time.time()

        # print(f"Time taken for ViT Inference: {t1-t0:.2f}s")

        logging.info(f"Time taken for ViT Inference: {t1-t0:.2f}s")
        
        return prediction