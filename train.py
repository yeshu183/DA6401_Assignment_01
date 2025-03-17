import argparse
import wandb
import numpy as np
import pickle
from dataset import load_data
from model import NN
from config import DEFAULT_CONFIG

# Argument Parser
parser = argparse.ArgumentParser(description="Train a neural network on MNIST or Fashion-MNIST")
parser.add_argument("--wandb_project", "-wp", type=str, default="example run")
parser.add_argument("--wandb_entity", "-we", type=str, default="yeshu183-indian-institute-of-technology-madras")
parser.add_argument("--dataset", "-d", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist")
parser.add_argument("--epochs", "-e", type=int, default=DEFAULT_CONFIG["epochs"])
parser.add_argument("--batch_size", "-b", type=int, default=DEFAULT_CONFIG["batch_size"])
parser.add_argument("--loss", "-l", type=str, choices=["mean_squared_error", "cross_entropy"], default=DEFAULT_CONFIG["loss"])
parser.add_argument("--optimizer", "-o", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default=DEFAULT_CONFIG["optimizer"])
parser.add_argument("--learning_rate", "-lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
parser.add_argument("--momentum", "-m", type=float, default=DEFAULT_CONFIG["momentum"])
parser.add_argument("--beta", "--beta1", "-beta", type=float, default=DEFAULT_CONFIG["beta"])
parser.add_argument("--beta2", "-beta2", type=float, default=DEFAULT_CONFIG["beta2"])
parser.add_argument("--epsilon", "-eps", type=float, default=DEFAULT_CONFIG["epsilon"])
parser.add_argument("--weight_decay", "-w_d", type=float, default=DEFAULT_CONFIG["weight_decay"])
parser.add_argument("--weight_init", "-w_i", type=str, choices=["random", "Xavier"], default=DEFAULT_CONFIG["weight_init"])
parser.add_argument("--num_layers", "-nhl", type=int, default=DEFAULT_CONFIG["num_layers"])
parser.add_argument("--hidden_size", "-sz", type=int, default=DEFAULT_CONFIG["hidden_size"])
parser.add_argument("--activation", "-a", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default=DEFAULT_CONFIG["activation"])
args = parser.parse_args()

# Initialize W&B
wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

# Load Data
x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)

# Initialize Model
model = NN(
    input_shape=x_train.shape[1],
    output_shape=10,
    n_hidden_layers=args.num_layers,
    h_per_layer=args.hidden_size,
    activation_func=args.activation,
    init_type=args.weight_init,
    loss_func=args.loss,
    optimizer_func=args.optimizer,
    learning_rate=args.learning_rate,
    l2_reg=args.weight_decay
)

# Train Model
train_loss, train_acc, val_loss, val_acc = model.train(x_train, y_train, x_val, y_val, args.epochs, args.batch_size)

# Evaluate Model
y_hat_test = model.predict(x_test)
final_test_acc = model.accuracy(y_hat_test, y_test)
final_test_loss = model.loss(y_hat_test, model.one_hot(y_test))

# Log Metrics
wandb.log({"final_test_loss": final_test_loss, "final_test_acc": final_test_acc})
print(f"Test Accuracy: {final_test_acc:.4f}")


wandb.finish()
