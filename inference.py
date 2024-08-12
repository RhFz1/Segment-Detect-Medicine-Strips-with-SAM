import torch
import torch.nn as nn
import warnings
from preprocess import data_transforms
from PIL import Image
from model import Model 

warnings.filterwarnings("ignore")

model_path = '/home/ec2-user/FAIR/SAM/registry/model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to(device)

# Function to perform inference
def predict(image_path: str):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Convert the matrix to a PIL image
    # image = Image.fromarray(image_matrix.astype('uint8'), 'RGB')
    
    # Apply transformations
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device    
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = output
    
    return prediction