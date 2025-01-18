import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from MLP.ThreeLayerMLP import NeuralNet  
import matplotlib.pyplot as plt


losses_lr1 = []
losses_lr2 = []
losses_mlp1 = []
losses_mlp2 = []

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # load image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images / 255.0  # Normalize pixel values to [0, 1]

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# load the training data
X_train = load_mnist_images('mnist/train-images.idx3-ubyte')
y_train = load_mnist_labels('mnist/train-labels.idx1-ubyte')

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

# Create dataset and data loader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# =================================================================
# Logistic Regression
# =================================================================
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

logistic_model = LogisticRegression(input_size=28*28, num_classes=10)

# Loss and optimizer for Logistic Regression
criterion = nn.CrossEntropyLoss()
optimizer_lr1 = torch.optim.SGD(logistic_model.parameters(), lr=0.01)  # SGD

# Training
num_epochs_lr = 10
for epoch in range(num_epochs_lr):
    for i, (images, labels) in enumerate(train_loader):
        outputs = logistic_model(images) # Forward pass
        loss = criterion(outputs, labels)  # calculate loss
        losses_lr1.append(loss.item())

        optimizer_lr1.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass to calculate gradients
        optimizer_lr1.step()  # Update parameters

        if (i+1) % 100 == 0:
            print(f'Logistic Regression - Epoch [{epoch+1}/{num_epochs_lr}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


torch.save(logistic_model.state_dict(), 'logistic_model1.pth')
print("Logistic Regression model saved as logistic_model1.pth")
# =================================================================
optimizer_lr2 = torch.optim.Adam(logistic_model.parameters(), lr=0.01)  # Adam
for epoch in range(num_epochs_lr):
    for i, (images, labels) in enumerate(train_loader):
        outputs = logistic_model(images) # Forward pass
        loss = criterion(outputs, labels)  # calculate loss
        losses_lr2.append(loss.item())

        optimizer_lr2.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass to calculate gradients
        optimizer_lr2.step()  # Update parameters

        if (i+1) % 100 == 0:
            print(f'Logistic Regression - Epoch [{epoch+1}/{num_epochs_lr}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


torch.save(logistic_model.state_dict(), 'logistic_model2.pth')
print("Logistic Regression model saved as logistic_model2.pth")

# =================================================================
# MLP
# =================================================================
input_size = 28 * 28
hidden_size = 500
num_epochs=10
num_classes =10
batch_size = 100
learning_rate = 0.001

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses_mlp1.append(loss.item())

        optimizer1.zero_grad()  
        loss.backward()  
        optimizer1.step()  

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model1.pth')

# =================================================================
optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses_mlp2.append(loss.item())

        # back propagation and optimizer
        optimizer2.zero_grad()  
        loss.backward()  
        optimizer2.step()  

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'model2.pth')

plt.figure(figsize=(12, 8))
plt.plot(losses_lr1, label='Logistic Regression - SGD', color='blue')
plt.plot(losses_lr2, label='Logistic Regression - Adam', color='orange')
plt.plot(losses_mlp1, label='MLP - Adam', color='green')
plt.plot(losses_mlp2, label='MLP - SGD', color='red')

plt.title('Loss Curves for Different Models')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('loss_curves.png')  
plt.show()  