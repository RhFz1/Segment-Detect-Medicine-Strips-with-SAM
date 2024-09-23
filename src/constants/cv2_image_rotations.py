import cv2
import numpy as np

def rotate_image(cropped_image, angle):
# Assuming cropped_image is the image you want to rotate
    (h, w) = cropped_image.shape[:2]

    # Compute the center of the image
    center = (w // 2, h // 2)

    # Rotation matrix (270 degrees counterclockwise = -90 degrees)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Get the sine and cosine of the rotation matrix
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image
    new_w = int((h * abs_sin) + (w * abs_cos))
    new_h = int((h * abs_cos) + (w * abs_sin))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and expand the image
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_w, new_h))

    return rotated_image
