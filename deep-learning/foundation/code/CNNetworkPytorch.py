import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# DATA INITIALIZATION PIPELINE
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

#CNN INTERNAL STRUCTURE (Model Architecture)
class HandwritingCNN(nn.Module):
    def __init__(self):
        super(HandwritingCNN, self).__init__()

        #Convolution model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*13*13, 128)
        self.relu2 = nn.ReLU()

        self.fc2 = nn.Linear(128,10)
        self.log_softmax = nn.LogSoftmax(dim=1)

        #dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        #Forward pass the data through the CNN layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        #Flatten from 3D to 1D for output
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        #Output layer
        x = self.fc2(x)
        x = self.log_softmax(x)

        return x

model = HandwritingCNN()

#Loss Function and Optimizer
criterion = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

#Training Loop (Important)

epochs = 10
print("Training Phase: ")
for epoch in range(epochs):
    total_loss = 0
    total_samples = 0
    correct_predictions = 0

    for images, labels in train_loader:

        #FEEDFORWARD
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        #BACKPROPOGATION
        optimizer.zero_grad() #clearing old gradients from the last batch
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 3. Calculate Accuracy
        # Find the index (0-9) of the highest probability
        winning_classes = torch.argmax(predictions, dim=1)
        
        # Tally up the total images and how many we got right
        total_samples += labels.size(0)
        correct_predictions += (winning_classes == labels).sum().item()
        
    # Calculate averages for the entire epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = (correct_predictions / total_samples) * 100
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")