import easyocr
import numpy as np
from PIL import Image, ImageFilter
from src.constants.cv2_image_rotations import rotate_image

def enhance_image_for_ocr(image_path, scale_factor=2.0):
    # Step 1: Enlarge the image
    # Step 1: Enlarge the image
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    enlarged_image = image.resize(new_size, Image.LANCZOS)

    # Step 3: Denoise the image (using OpenCV's fastNlMeansDenoising for noise reduction)
    open_cv_image = np.array(enlarged_image)
    # denoised_image = cv2.fastNlMeansDenoising(open_cv_image, None, 30, 7, 21)

    # Step 4: Sharpen the image using PIL's filter
    pil_denoised_image = Image.fromarray(open_cv_image)
    sharpened_image = pil_denoised_image.filter(ImageFilter.SHARPEN)

    sharpened_image.save('./assets/enhanced_image.jpg')

    # Save or return the final enhanced image
    return sharpened_image

# Example usage
enlarged_image = enhance_image_for_ocr('./assets/cropped_image_2.jpg', 2)

ocr = easyocr.Reader(['en'])

for rotation in range(0, 360, 90):
    rotated_image = rotate_image(np.array(enlarged_image), rotation)
    print(ocr.readtext(image=rotated_image, detail = 0, paragraph = True))
