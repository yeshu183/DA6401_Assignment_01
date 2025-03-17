# **DA6401 - Assignment 1**
### **Fundamentals of Deep Learning**

This repository contains the implementation of a **fully connected neural network (NN)** for the **MNIST** and **Fashion-MNIST** datasets. The project includes training, evaluation, hyperparameter tuning with WandB sweeps, and logging of results.


## 📂 **Project Structure**
```bash
.
├── best_model.py         # Loads and evaluates the best trained model
├── config.py             # Default hyperparameters configuration
├── dataset.py            # Data loading and preprocessing functions
├── model.py              # Neural network model implementation
├── optimizers.py         # Various optimization algorithms (SGD, Adam, etc.)
├── requirements.txt      # Required dependencies
├── sweep.py              # Hyperparameter tuning using WandB sweeps
├── train.py              # Training script with command-line arguments
└── README.md             # This file (Project Documentation)
```

---

## 🛠 **Setup Instructions**
### **1️⃣ Install Dependencies**
Run the following command to install the required dependencies:
```bash
pip install -r requirements.txt
```
Dependencies include:
- `numpy`
- `matplotlib`
- `seaborn`
- `wandb`
- `keras`
- `tensorflow`

### **2️⃣ Running the Training Script**
To train the model, run:
```bash
python train.py --wandb_project "my_project" --wandb_entity "my_username"
```
#### **Supported Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--wandb_project` | WandB project name | `"example run"` |
| `--wandb_entity` | WandB entity name | `"yeshu183-indian-institute-of-technology-madras"` |
| `--dataset` | Choose dataset: `mnist` or `fashion_mnist` | `"fashion_mnist"` |
| `--epochs` | Number of training epochs | `10` |
| `--batch_size` | Batch size for training | `32` |
| `--loss` | Loss function: `mean_squared_error`, `cross_entropy` | `"cross_entropy"` |
| `--optimizer` | Optimizer: `sgd`, `adam`, `rmsprop`, etc. | `"adam"` |
| `--learning_rate` | Learning rate | `0.001` |
| `--momentum` | Momentum (for SGD/NAG) | `0.9` |
| `--beta1` | Beta1 (for Adam/Nadam) | `0.9` |
| `--beta2` | Beta2 (for Adam/Nadam) | `0.999` |
| `--epsilon` | Epsilon (for numerical stability) | `1e-8` |
| `--weight_decay` | Weight decay (L2 regularization) | `0.0` |
| `--weight_init` | Weight initialization: `random`, `xavier` | `"xavier"` |
| `--num_layers` | Number of hidden layers | `3` |
| `--hidden_size` | Number of neurons per hidden layer | `64` |
| `--activation` | Activation function: `sigmoid`, `tanh`, `ReLU` | `"relu"` |

---

## 🎯 **Training the Neural Network**
- The neural network is implemented in `model.py` using the `NN` class.
- Model initialization:
```python
from model import NN
from optimizers import Optimizers

optimizer = Optimizers(lr=0.001, optimizer="adam")
model = NN(input_shape=784, output_shape=10, n_hidden_layers=3, h_per_layer=64,
           activation_func="relu", loss_func="cross_entropy_loss",
           init_type="xavier", optimizer=optimizer)
```
- Training:
```python
train_loss, train_acc, val_loss, val_acc = model.train(x_train, y_train, x_val, y_val, epochs=10, batch_size=32)
```

---

## 📊 **Hyperparameter Tuning with WandB Sweeps**
To run hyperparameter tuning using **WandB sweeps**, execute:
```bash
python sweep.py
```
- Uses **Bayesian Optimization** to find the best hyperparameters.
- The best model is saved as `"best_model.pkl"` in WandB.

---

## 🏆 **Best Model & Evaluation**
Once the best model is trained, load and evaluate it:
```bash
python best_model.py
```
- Downloads the best model from **WandB**.
- Computes **test accuracy** and **confusion matrix**.

---

## 📊 **Confusion Matrix Logging**
The confusion matrix is **logged to WandB** using:
```python
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
```
### **📌 Custom Confusion Matrix**
Instead of using WandB's default confusion matrix, a **custom confusion matrix** is plotted using Matplotlib and Seaborn:
```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Custom Confusion Matrix")
plt.savefig("confusion_matrix.png")
wandb.log({"Custom Confusion Matrix": wandb.Image("confusion_matrix.png")})
```
- ✅ **Color-coded true/false predictions** using **Greens/Reds**.
- ✅ **Percentage labels instead of raw counts**.

---

## 🔥 **Results & Best Hyperparameters**
After running hyperparameter tuning, these were the **best-performing hyperparameters**:

| **Hyperparameter** | **Best Value** |
|-------------------|-------------|
| `optimizer` | `"adam"` |
| `learning_rate` | `0.001` |
| `hidden_size` | `64` |
| `batch_size` | `32` |
| `num_layers` | `3` |
| `activation` | `"relu"` |
| `weight_init` | `"xavier"` |

**Test Accuracy on Fashion-MNIST:**  
```
Final Test Accuracy: 89.7%
```

---

## 📜 **Conclusions**
- **Cross-Entropy Loss** was more stable than Squared Error Loss.
- **Adam optimizer** performed the best, avoiding gradient explosion.
- **Weight initialization (Xavier)** prevented unstable activations.
- The model achieved **high accuracy** with a **custom confusion matrix visualization**.

---
