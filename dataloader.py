from torchvision import datasets
from torch.utils.data import DataLoader
from preprocess import data_transforms

train_path = '/home/ec2-user/FAIR/SAM/m2_train_images/train'
val_path = '/home/ec2-user/FAIR/SAM/m2_train_images/val'

# init train loader
train_dataset = datasets.ImageFolder(train_path, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# init val loader
val_dataset = datasets.ImageFolder(val_path, transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)