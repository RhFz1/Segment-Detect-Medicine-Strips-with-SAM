import os
import time
import cv2
import io
from src.logging.logger import logging
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
from PIL import Image


load_dotenv('.env')

class OCR():
    @staticmethod
    def perform_ocr(cv2_image):
        """
        Perform OCR on an image using Azure's Computer Vision API
        
        Args:
        
        Returns:
            str: Extracted text from the image
        """
        try:
            # Create an authenticated client
            credentials = CognitiveServicesCredentials(os.getenv('AZURE_KEY'))
            client = ComputerVisionClient(endpoint=os.getenv('AZURE_ENDPOINT'), credentials=credentials)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2_image)
            
            # Create a byte stream to hold the image data
            image_stream = io.BytesIO()
            pil_image.save(image_stream, format='JPEG')
            image_stream.seek(0)

            read_response = client.read_in_stream(image_stream, raw=True)

            # Get the operation location (URL with ID at the end)
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            # Wait for the operation to complete
            while True:
                read_result = client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(0.5)

            # Extract the text
            if read_result.status == OperationStatusCodes.succeeded:
                text = ""
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        text += line.text + "\n"
                return text.strip()
            else:
                return f"OCR operation failed with status: {read_result.status}"

        except Exception as e:
            logging.error(f"Error in performing OCR: {str(e)}")
            return f"Error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Your Azure credentials
    subscription_key = "xxxxxxxxxxxxxxxxxxxxxxxxx"
    endpoint = "xxxxxxxxxxxxxxxxxxxxxxx"
    
    # Path to your image
    image_path = "runs/cropped_strip_1.jpeg"
    image = cv2.imread(image_path)
    # Perform OCR
    extracted_text = OCR.perform_ocr(subscription_key, endpoint,image)
    print("Extracted text:")
    print(extracted_text)