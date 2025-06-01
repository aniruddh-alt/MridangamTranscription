import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MridangamCNN(nn.Module):
    def __init__(self, n_mels = 128, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(

            # input shape: (1, 128, 128)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (32, 128, 128)

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (32, 64, 64)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (64, 64, 64)

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (64, 32, 32)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1),
            # output shape: (128, 32, 32)

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            # output shape: (128, 16, 16)

            nn.AdaptiveAvgPool2d((1, None)) # output shape: (128, 1, 16)

        )

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),  # output shape: (64, 1, 16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # output shape: (64, 1, 16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.AdaptiveAvgPool1d(1),  # output shape: (64, 1, 1)
            nn.Flatten(),  # output shape: (64)
            nn.Linear(64, num_classes),  # output shape: (num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2)
        x = self.classifier(x)
        return x

# Training and Evaluation
model = MridangamCNN(n_mels=128, num_classes=10)  # Adjust num_classes as needed
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return losses

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    losses = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            losses.append(loss.item())
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    avg_loss = sum(losses) / len(losses)
    print(f'Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}')
    return accuracy, avg_loss

