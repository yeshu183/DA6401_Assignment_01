import argparse
import wandb
import numpy as np
import pickle
from dataset import load_dataset
from model import NN
from config import DEFAULT_CONFIG
from optimizers import Optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
# Connect to W&B
api = wandb.Api()

# Replace with your project name
project_name = "New Sweep"
entity_name = wandb.Api().default_entity

# Get all runs
runs = api.runs(f"{entity_name}/{project_name}")

# Find the best run based on `final_val_acc`
best_run = max(runs, key=lambda run: run.summary.get("final_val_acc", 0))

print(f"Best run ID: {best_run.id}, Accuracy: {best_run.summary.get('final_val_acc', 0)}")

# Download the best model
best_run.file("best_model.pkl").download(replace=True)
print("Best model downloaded successfully!")

# Load the saved model
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

print("Best model loaded successfully!")

x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("fashion_mnist")

# Initialize W&B for testing
wandb.init(project="fashion-mnist", name="custom-confusion-matrix")

# Run inference on the test set
a_list_test, h_list_test = best_model.forward(x_test)
y_hat_test = h_list_test[best_model.n_h+1]

# Compute test loss and accuracy
final_test_loss = best_model.loss(y_hat_test, best_model.one_hot(y_test))
final_test_acc = best_model.accuracy(np.argmax(np.array(y_hat_test), axis=1), y_test)

print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.4f}")

class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
]

# Compute confusion matrix
cm = confusion_matrix(y_test, np.argmax(np.array(y_hat_test),axis=1))
mask = np.eye(len(cm), dtype=bool)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Greens",mask=~mask,xticklabels=class_labels,yticklabels=class_labels,cbar=False)
sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Reds",mask=mask,xticklabels=class_labels,yticklabels=class_labels,cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Custom Confusion Matrix")

# Save the figure
plt.savefig("confusion_matrix.png", bbox_inches="tight")
wandb.log({"Custom Confusion Matrix": wandb.Image("confusion_matrix.png")})

# Close wandb
wandb.finish()
