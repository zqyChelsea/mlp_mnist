import numpy as np
import struct
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from MLP.ThreeLayerMLP import NeuralNet

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images / 255.0  

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


X_test = load_mnist_images('mnist/t10k-images.idx3-ubyte')
y_test = load_mnist_labels('mnist/t10k-labels.idx1-ubyte')

X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# load test data
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# =================================================================
# test the Logistic Regression model

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model and load weights
logistic_model1 = LogisticRegression(input_size=28*28, num_classes=10)
logistic_model1.load_state_dict(torch.load('logistic_model1.pth')) # Load trained weights
logistic_model1.eval()  # Set model to evaluation mode

correct = 0
total = 0
with torch.no_grad(): # Disable gradient calculation for inference
    for images, labels in test_loader:
        outputs = logistic_model1(images)
        _, predicted = torch.max(outputs.data, 1) # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # Count correct predictions

print(f'Accuracy of Logistic Regression model 1 on the test images: {100 * correct / total:.2f}%')

# =================================================================
logistic_model2 = LogisticRegression(input_size=28*28, num_classes=10)
logistic_model2.load_state_dict(torch.load('logistic_model2.pth')) # Load trained weights
logistic_model2.eval()  # Set model to evaluation mode

correct = 0
total = 0
with torch.no_grad(): # Disable gradient calculation for inference
    for images, labels in test_loader:
        outputs = logistic_model2(images)
        _, predicted = torch.max(outputs.data, 1) # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # Count correct predictions

print(f'Accuracy of Logistic Regression model 2 on the test images: {100 * correct / total:.2f}%')

# =================================================================
# MLP model 1
input_size = 28 * 28
hidden_size = 500
num_classes = 10

mlp_model_1 = NeuralNet(input_size, hidden_size, num_classes)
mlp_model_1.load_state_dict(torch.load('model1.pth'))
mlp_model_1.eval()  

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = mlp_model_1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the MLP model 1 on the test images: {100 * correct / total:.2f}%')

# =================================================================
# MLP model 2

mlp_model_2 = NeuralNet(input_size, hidden_size, num_classes)
mlp_model_2.load_state_dict(torch.load('model2.pth'))
mlp_model_2.eval()  

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = mlp_model_2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the MLP model 2 on the test images: {100 * correct / total:.2f}%')