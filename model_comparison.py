import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

original_acc = 100 * np.load('test_acc_original.npy')
original_mse = np.load('test_mse_original.npy')
original_auc = np.load('test_auc_original.npy')

original_train_mse = np.load('epoch_loss_original.npy')
original_train_acc = np.load('epoch_train_acc_original.npy')

modified_acc = 100 * np.load('test_acc_modified.npy')
modified_mse = np.load('test_mse_modified.npy')
modified_auc = np.load('test_auc_modified.npy')

modified_train_mse = np.load('epoch_loss_modified.npy')
modified_train_acc = np.load('epoch_train_acc_modified.npy')

pytorch_train_acc = np.load('train_acc_pytorch.npy')
pytorch_train_loss = np.load('epoch_loss_pytorch.npy')
pytorch_test_acc = np.load('test_acc_pytorch.npy')
pytorch_test_auc = np.load('test_auc_pytorch.npy')

ch11_epochs = len(original_train_acc)
pytorch_epochs = len(pytorch_train_acc)

# Plot training accuracy
plt.plot(np.arange(1, pytorch_epochs+1), pytorch_train_acc, label='PyTorch')
plt.plot(np.arange(1, ch11_epochs+1), original_train_acc, label='Original')
plt.plot(np.arange(1, ch11_epochs+1), modified_train_acc, label='Modified')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Plot training loss
plt.plot(np.arange(1, ch11_epochs+1), original_train_mse, label='Original')
plt.plot(np.arange(1, ch11_epochs+1), modified_train_mse, label='Modified')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.legend()
plt.grid()
plt.show()

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'Model': ['PyTorch', 'Original', 'Modified'],
    'Epochs': [pytorch_epochs, ch11_epochs, ch11_epochs],
    'Test Accuracy (%)': [np.round(pytorch_test_acc,3), np.round(original_acc,3), np.round(modified_acc,3)],
    'Test AUC': [np.round(pytorch_test_auc,3), np.round(original_auc,3), np.round(modified_auc,3)]
})
# Plot the table
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=results.values, colLabels=results.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.title('Model Comparison')
plt.show()