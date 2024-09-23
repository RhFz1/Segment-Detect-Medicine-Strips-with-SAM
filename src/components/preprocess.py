from torchvision import transforms

# Data transformations
data_transforms = transforms.Compose([
    transforms.ToPILImage(),  # Convert the input numpy array to a PIL image
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),  # Convert to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
])