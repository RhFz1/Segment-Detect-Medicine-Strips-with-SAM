import cv2

def enlarge_image(input_image, scale_factor):
    # Get the current dimensions of the image
    height, width = input_image.shape[:2]
    
    # Calculate new dimensions
    new_size = (int(width * scale_factor), int(height * scale_factor))
    
    # Resize the image
    enlarged_img = cv2.resize(input_image, new_size, interpolation=cv2.INTER_LINEAR)
    return enlarged_img