import torch   
import os
import warnings
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from src.logging.logger import logging
from src.config.configuration import ImageConfig
from src.utils.image_handler import crop_image, enhance_image
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = os.path.join("./artifacts/sam", "sam_vit_h.pth")

class SAM():
    def __init__(self):
        self.sam = SamPredictor(sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE))
        self.data_config = ImageConfig()
    def set_image(self, image):
        """
            Set the image in the SAM model.

            Args:
                image (np.ndarray): The image.
        """

        self.sam.set_image(image)
    def unset_image(self):
        self.sam.reset_image()
    def generate_map(self, coordinates):
        """
            Generate a map of the coordinates using the SAM model.

            Args:
                sam (SAM): The SAM model.
                image ([np.ndarray]): The image.
                coordinates ([np.ndarray]): The list of coordinates.

            Returns:
                [np.ndarray]: The list of maps.
        """
        try:
            masks, _, box = self.sam.predict(
                point_coords=None,
                point_labels=None,
                box=np.array([coordinates]),
                multimask_output=False
            )
            # Retreive the mask
            mask = masks[0]
            # Getting the pixel area of the mask.
            area = np.sum(mask)
            # Save the segmented area
            return mask, area, box
        except Exception as e:
            logging.error(f"Error in generating map: {str(e)}")
            return None, None, None
    
    def get_scale_factor(self, coordinates):
        """
            Get the scale factor.

            Returns:
                float: The scale factor.
        """
        _, area, _ = self.generate_map(coordinates)
        return self.data_config.area_real / area
    
    def get_cropped_image(self, image, coordinates):
        """
            Get the cropped image.

            Args:
                image ([np.ndarray]): The image.
                mask ([np.ndarray]): The mask.

            Returns:
                [np.ndarray]: The cropped image.
                int: Strip pixel area.
        """

        # Generating the map
        # Along with the strip pixel area
        mask, strip_pixel_area, _ = self.generate_map(coordinates)

        # Now we need to crop the image
        # This is done to boost OCR performance
        cropped_image = crop_image(image, mask, self.data_config.pix_scale)

        # Check if the cropped image is of low resolution
        # Again this is to boost OCR performance
        enhanced_image = enhance_image(cropped_image, self.data_config.img_scale)

        return enhanced_image, strip_pixel_area