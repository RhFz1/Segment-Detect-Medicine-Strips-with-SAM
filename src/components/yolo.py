import cv2
import numpy as np
from ultralytics import YOLO


class Yolo():
    def __init__(self, confidence_thrshold=0.75):
        self.model_weights_path = "./artifacts/yolo/yolov2.pt"
        self.model = YOLO(self.model_weights_path)
        self.confidence_threshold = confidence_thrshold
        self.y_offset = 30
        self.x_offset = 30
        self.map = {0: 'box', 1: 'circle', 2: 'strip'}

    def inference(self, image):
        
        res = {'strip': [], 'circle': [], 'box': []}

        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image = image
        else:
            raise ValueError("Invalid image format")

        count = 0
        # Perform object detection
        results = self.model(image, save=False)  # Save the image with bounding boxes
        
        # Access the first item in results to get detection details
        detections = results[0].boxes  # This gives you access to the bounding boxes
        #results[0].show()    
        
        for box in detections:
            # Here we pull bounding boxes and the class label of either circle or strip.
            # box contains [x1, y1, x2, y2, confidence, class_label]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            confidence = box.conf[0].item()  # Confidence score
            class_label = int(box.cls[0].item())  # Class label (0 for strip, 1 for circle)	

            # If the confidence is greater than the threshold, we add the bounding box to the respective list.
            if confidence > self.confidence_threshold:
                res[self.map[class_label]].append(np.array([x1, y1, x2, y2]))
        return res

if __name__ == '__main__':
    strips = Yolo()
    #for i in range(1, 30):
    strip_no = strips.inference(f'./assets/new_med.jpg')
    print(strip_no)