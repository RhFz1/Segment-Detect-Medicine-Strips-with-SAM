import cv2
import numpy as np
from ultralytics import YOLO


class StripCount():
    def __init__(self,confidence_thrshold=0.75):
        self.model_weights_path="./artifacts/yolo/yolo.pt"
        self.model=YOLO(self.model_weights_path)
        self.confidence_threshold=confidence_thrshold
    def calculate_strip_count(self,image):
        if isinstance(image,str):
             image = cv2.imread(image)
        elif isinstance(image,np.ndarray):
             image=image
        else :
            raise ValueError("Invalid image format")
        count=0
        # Perform object detection
        results = self.model(image, save=False)
        for result in results:
            #result.show()
            boxes=result.boxes
            for box in boxes:
                if box.conf>self.confidence_threshold:
                     count+=1
        return count

if __name__=='__main__':
    strips=strip_count()
    strip_no=strips.calculate_strip_count("assests/conf_testing/img_006.jpg")
    print(f"no of strips:{strip_no}")

        



