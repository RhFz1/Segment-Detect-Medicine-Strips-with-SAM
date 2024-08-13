import os 
import shutil


HOME = "/home/ec2-user/FAIR/SAM"

# for i, img in enumerate(os.listdir(os.path.join(HOME, "images"))):
#     ext = img.split('.')[-1]
#     source = os.path.join(HOME, "images", img)
#     destination = os.path.join(HOME, 'images', f'example_image_{i + 1}.{ext}')
#     shutil.move(source, destination)
#     print('\r'+ f"Moved {source} to {destination}", end = '')

print()
train_dir = os.path.join(HOME, "m2_train_images")
for i, img in enumerate(os.listdir(os.path.join(train_dir, 'train','No'))):
    ext = img.split('.')[-1]
    if (i + 1) % 15 == 1:
        source = os.path.join(train_dir, 'train','No', img)
        destination = os.path.join(train_dir, 'val', 'No', f'No_{i + 1}.{ext}')
        shutil.move(source, destination)
        print('\r'+ f"Moved {source} to {destination}", end = '')

for i, img in enumerate(os.listdir(os.path.join(train_dir, 'train','Yes'))):
    ext = img.split('.')[-1]
    if (i + 1) % 15 == 1:
        source = os.path.join(train_dir, 'train','Yes', img)
        destination = os.path.join(train_dir, 'val', 'Yes', f'Yes_{i + 1}.{ext}')
        shutil.move(source, destination)
        print('\r'+ f"Moved {source} to {destination}", end = '')