import os 
import shutil


HOME = "/home/ec2-user/FAIR/SAM"

for i, img in enumerate(os.listdir(os.path.join(HOME, "images"))):
    ext = img.split('.')[-1]
    source = os.path.join(HOME, "images", img)
    destination = os.path.join(HOME, 'images', f'example_image_{i + 1}.{ext}')
    shutil.move(source, destination)
    print('\r'+ f"Moved {source} to {destination}", end = '')