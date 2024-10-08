import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import Model
from dataloader import train_loader, train_dataset, val_loader, val_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Create model, define loss function and optimizer
model = Model().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    preds = outputs >= 0.5
    corrects = torch.sum(preds == labels.byte())
    return corrects.double() / len(labels)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    start_time = time.time()
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += calculate_accuracy(outputs, labels) * inputs.size(0)
    

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)

    val_loss = 0.0
    val_corrects = 0
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            
            val_loss = 0.0
            val_corrects = 0
            
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += calculate_accuracy(outputs, labels) * inputs.size(0)
        
        eval_loss = val_loss / len(val_dataset)
        eval_acc = val_corrects / len(val_dataset)
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_loss:.4f}, Train accuracy: {epoch_acc:.4f}, Val loss: {eval_loss:.4f}, Val accuracy: {eval_acc: .4f} ,Time: {epoch_time:.4f}s')
    else:
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_time:.4f}s')

    # Step the scheduler
    scheduler.step()
    
    
# Save the model
torch.save(model.state_dict(), os.path.join('/home/ec2-user/FAIR/SAM/registry', 'model_lat.pth'))