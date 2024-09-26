import cv2
import numpy as np
from ultralytics import YOLO


class StripCount():
    def __init__(self, confidence_thrshold=0.75):
        self.model_weights_path = "./artifacts/yolo/yolo.pt"
        self.model = YOLO(self.model_weights_path)
        self.confidence_threshold = confidence_thrshold
        self.y_offset = 30
        self.x_offset = 30

    def inference(self, image):
        
        res = {'strip': [], 'circle': []}

        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image = image
        else:
            raise ValueError("Invalid image format")

        count = 0
        # Perform object detection
        results = self.model(image, save=False)
        
        # Access the first item in results to get detection details
        detections = results[0].boxes  # This gives you access to the bounding boxes    
        
        for box in detections:
            # Here we pull bounding boxes and the class label of either circle or strip.
            # box contains [x1, y1, x2, y2, confidence, class_label]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            confidence = box.conf[0].item()  # Confidence score
            class_label = int(box.cls[0].item())  # Class label (0 for strip, 1 for circle)

            # If the confidence is greater than the threshold, we add the bounding box to the respective list.
            if confidence > self.confidence_threshold:
                res['circle' if class_label == 0 else 'strip'].append(np.array([x1, y1, x2, y2]))
        
        return res

if __name__ == '__main__':
    strips = StripCount()
    i = 4
    strip_no = strips.inference(f"./assets/img_00{i}.jpg")
    print(strip_no)
