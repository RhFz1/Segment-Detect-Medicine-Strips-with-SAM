import numpy as np
import cv2
import io
from PIL import Image
from scipy.spatial import ConvexHull

def read_image(image_path: str = None, image: Image = None) -> np.ndarray:
    """
    Reads the image from the given path or PIL image object

    Args:
    image_path (str): Path to the image
    image (PIL.Image): PIL Image object

    Returns:
    np.ndarray: Image as a numpy array
    """
    try:
        if image_path:
            # Check if the image path is a string
            if not isinstance(image_path, str):
                raise TypeError("image_path should be a string")
            # Read the image using OpenCV
            img = cv2.imread(image_path)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            return img
        elif image:
            # Check if the image is a PIL Image object
            if not isinstance(image, Image.Image):
                raise TypeError("image should be a PIL Image object")
            return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Either image_path or image should be provided")
    except Exception as e:
        print(f"Error reading image: {e}")
        raise

def crop_image(image: np.ndarray, mask: np.ndarray, pix_thresh: int) -> np.ndarray:
    """
    Crops the image to the given coordinates

    Args:
    image (np.ndarray): Image as a numpy array
    mask (np.ndarray): Mask to crop the image

    Returns:
    np.ndarray: Cropped image
    """
    try:
        # Check if the inputs are numpy arrays
        if not isinstance(image, np.ndarray):
            raise TypeError("image should be a numpy array")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask should be a numpy array")
        
        # Find the bounding box of the mask (non-black region)
        y_indices, x_indices = np.where(mask)  # Get indices of the True values in the mask
        if y_indices.size == 0 or x_indices.size == 0:
            raise ValueError("Mask does not contain any True values")
        
        x_min, x_max = x_indices.min(), x_indices.max()  # Min and Max X coordinates
        y_min, y_max = y_indices.min(), y_indices.max()  # Min and Max Y coordinates

        # Apply pixel threshold to the bounding box coordinates
        x_min = max(x_min - pix_thresh, 0)
        x_max = min(x_max + pix_thresh, image.shape[1])
        y_min = max(y_min - pix_thresh, 0)
        y_max = min(y_max + pix_thresh, image.shape[0])

        return image[y_min:y_max, x_min:x_max]
    except Exception as e:
        print(f"Error cropping image: {e}")
        raise

def rotate_image(cropped_image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotates the image by the given angle

    Args:
    cropped_image (np.ndarray): Cropped image as a numpy array
    angle (int): Angle to rotate the image

    Returns:
    np.ndarray: Rotated image
    """
    try:
        # Check if the input is a numpy array
        if not isinstance(cropped_image, np.ndarray):
            raise TypeError("cropped_image should be a numpy array")
        # Check if the angle is an integer
        if not isinstance(angle, int):
            raise TypeError("angle should be an integer")
        
        (h, w) = cropped_image.shape[:2]

        # Compute the center of the image
        center = (w // 2, h // 2)

        # Rotation matrix
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
    except Exception as e:
        print(f"Error rotating image: {e}")
        raise

def align_image(image: np.ndarray,mask) -> np.ndarray:
    # Step 1: Get the indices of the ones in the mask
    try:
        # Check if the input is a numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError("input_image should be a numpy array")
        
        y_indices, x_indices = np.nonzero(mask)
        points = np.column_stack((x_indices, y_indices))  # Combine x and y indices into coordinates
    
        if points.shape[0] < 3:
            # Not enough points to form a polygon
            raise ValueError("Not enough points in mask to determine corners.")
    
        # Step 2: Compute the convex hull of the points
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    
        # Step 3: Find the minimum area rectangle
        min_area = None
        min_rect = None
    
        for i in range(len(hull_points)):
            # Get edge defined by hull points i and i+1
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
    
            # Compute the angle of the edge with respect to the x-axis
            angle = -np.arctan2(edge[1], edge[0])
    
            # Rotate all points by this angle
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            rotated_points = np.dot(points, rotation_matrix.T)
    
            # Get the bounding box of the rotated points
            min_x = np.min(rotated_points[:, 0])
            max_x = np.max(rotated_points[:, 0])
            min_y = np.min(rotated_points[:, 1])
            max_y = np.max(rotated_points[:, 1])
    
            area = (max_x - min_x) * (max_y - min_y)
    
            if (min_area is None) or (area < min_area):
                min_area = area
                min_rect = {
                    'angle': angle,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'rotation_matrix': rotation_matrix
                }
    
        # Step 4: Compute the corner points of the minimal rectangle
        rect_corners = np.array([
            [min_rect['min_x'], min_rect['min_y']],
            [min_rect['max_x'], min_rect['min_y']],
            [min_rect['max_x'], min_rect['max_y']],
            [min_rect['min_x'], min_rect['max_y']]
        ])
    
        # Rotate corners back to the original coordinate system
        inverse_rotation = np.linalg.inv(min_rect['rotation_matrix'])
        original_corners = np.dot(rect_corners, inverse_rotation.T)
    
        corners = np.round(original_corners).astype(int)
    
        edge1 = np.array(corners[1]) - np.array(corners[0])
        edge2 = np.array(corners[2]) - np.array(corners[1])
        
        if np.linalg.norm(edge1) > np.linalg.norm(edge2):
            angle = np.arctan2(edge1[1], edge1[0])
        else:
            angle = np.arctan2(edge2[1], edge2[0])
        
        angle_deg = np.degrees(angle)
        
        if angle_deg < -45:
            angle_deg += 90
        elif angle_deg > 45:
            angle_deg -= 90
    
        angle = angle_deg
    
        # Get the image size
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Step 5: Adjust rotation matrix to expand canvas size
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Compute the new bounding dimensions of the rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix to account for the translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform the rotation with the adjusted dimensions
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        print(f"Error aligning image: {e}")
        raise


def enlarge_image(input_image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Enlarges the image by the given scale factor

    Args:
    input_image (np.ndarray): Input image as a numpy array
    scale_factor (float): Scale factor to enlarge the image

    Returns:
    np.ndarray: Enlarged image
    """
    try:
        # Check if the input is a numpy array
        if not isinstance(input_image, np.ndarray):
            raise TypeError("input_image should be a numpy array")
        # Check if the scale factor is a float
        if not isinstance(scale_factor, (float, int)):
            raise TypeError("scale_factor should be a float or int")
        if scale_factor <= 0:
            raise ValueError("scale_factor should be greater than 0")
        
        # Get the current dimensions of the image
        height, width = input_image.shape[:2]
        
        # Calculate new dimensions
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # Resize the image
        enlarged_img = cv2.resize(input_image, new_size, interpolation=cv2.INTER_LINEAR)
        return enlarged_img
    except Exception as e:
        print(f"Error enlarging image: {e}")
        raise
def enhance_image(image: np.ndarray, scale: float) -> np.ndarray:
    """
    In case of a small image, it enlarges the image by the given scale factor

    Args:
    image (np.ndarray): Input image
    scale (float): Scale factor to enlarge the image

    Returns:
    np.ndarray: Enlarged image
    """

    height, width = image.shape[:2]

    if height < 400 or width < 400:
        image = enlarge_image(image, scale_factor=scale)
    
    return image

def image_to_bytes(image: Image):
    """
    Converts a PIL Image to bytes

    Args:
    image (PIL.Image): Input image

    Returns:
    bytes: Image as bytes
    """
    try:
        # Check if the input is a PIL Image
        if not isinstance(image, Image.Image):
            raise TypeError("image should be a PIL Image object")
        
        # Create a BytesIO object to store the image
        img_byte_array = io.BytesIO()
        
        # Save the image to the BytesIO object
        image.save(img_byte_array, format=image.format)
        
        # Get the value of the BytesIO buffer
        img_byte_array = img_byte_array.getvalue()
        return img_byte_array
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        raise
