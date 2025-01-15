import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

# normalize the data
X = ((X / 255.) - .5) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.relu(x)
        x = self.layers[-1](x)    
        return x

hidden_layers = [50, 25]
model = NeuralNet(input_size=28*28, hidden_sizes=hidden_layers, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
epoch_loss = []
epoch_train_acc = []
train_acc = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = 100 * correct / total
    epoch_loss.append(train_loss)
    epoch_train_acc.append(train_acc)
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')

print(f'Training accuracy: {train_acc:.2f}%')
plt.plot(np.arange(1, epochs+1), epoch_train_acc)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.show()

plt.plot(np.arange(1, epochs+1), epoch_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.show()


model.eval()
y_true = []
y_prob = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        y_true.append(y_batch.numpy())
        y_prob.append(probabilities.numpy())
        y_pred.append(predicted.numpy())


y_true = np.concatenate(y_true)
y_prob = np.concatenate(y_prob)
y_pred = np.concatenate(y_pred)

encoder = OneHotEncoder(sparse_output=False)
y_true_onehot = encoder.fit_transform(y_true.reshape(-1, 1))
test_auc = roc_auc_score(y_true_onehot, y_prob, average='macro', multi_class='ovr')
test_s_acc = 100 * accuracy_score(y_true, y_pred)
print(f'Test AUC: {test_auc:.3f}')
print(f'Test Accuracy: {test_s_acc:.2f}%')

np.save('train_acc_pytorch.npy', epoch_train_acc)
np.save('epoch_loss_pytorch.npy', epoch_loss)
np.save('test_acc_pytorch.npy', test_s_acc)
np.save('test_auc_pytorch.npy', test_auc)
