import wandb
import pickle
import numpy as np
from dataset import load_dataset
from model import NN
from optimizers import Optimizers
from dataset import load_dataset

# Define Sweep Configuration
sweep_config = {
    'method': 'bayes',  # Options: grid, random, bayes
    'metric': {'name': 'final_val_acc', 'goal': 'maximize'},
    'parameters': {
        'loss_func': {'values': ['cross_entropy_loss','squared_error']},#, 'squared_error'
        'epochs': {'values': [2,3]},
        'num_hid_layers': {'values': [3, 4, 5]},
        'hid_layer_size': {'values': [32,64,128]},
        'lr': {'values': [1e-3, 1e-4]},
        'weight_init': {'values': ['random', 'xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
        'l2_reg': {'values': [0, 0.0005, 0.5]},
        'batch_size': {'values': [16, 32, 64]},
        'optimizer_func': {'values': ['sgd','momentum', 'nesterov', 'rmsprop', 'adam']}
    }
}
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("fashion_mnist")
best_val_acc = 0
sweep_id = wandb.sweep(sweep_config, project="New Sweep")
def train_sweep():
    global best_val_acc
    wandb.init(project="New Sweep")
    config = wandb.config
    wandb.run.name = f"e_{config.epochs}_hl_{config.num_hid_layers}_nhpl_{config.hid_layer_size}_bs_{config.batch_size}_init_{config.weight_init}_ac_{config.activation}_op_{config.optimizer_func}_lo_{config.loss_func}"
    optimizer = Optimizers(lr=config.lr, beta1=0.9, beta2=0.999, rho=0.9, optimizer=config.optimizer_func)
    nn = NN(
        input_shape=784,
        n_hidden_layers=config.num_hid_layers,
        h_per_layer=config.hid_layer_size,
        activation_func=config.activation,
        init_type=config.weight_init,
        l2_reg=config.l2_reg,
        optimizer=optimizer,
        loss_func=config.loss_func,
        output_shape=10
    )

    train_loss_list,train_acc_list,val_loss_list,val_acc_list = nn.train(x_train, y_train, x_val, y_val, epochs=config.epochs, batch_size=config.batch_size)

    a_list_val,h_list_val = nn.forward(x_val)
    y_hat_val = h_list_val[nn.n_h+1]
    final_val_loss = nn.loss(y_hat_val,nn.one_hot(y_val))
    final_val_acc = nn.accuracy(np.argmax(np.array(y_hat_val),axis=1),y_val)
    wandb.log({"train_loss": train_loss_list})
    wandb.log({"train_acc": train_acc_list})
    wandb.log({"val_loss": val_loss_list})
    wandb.log({"val_acc": val_acc_list})
    wandb.log({"final_val_loss": final_val_loss})
    wandb.log({"final_val_acc": final_val_acc})
    print(f"Validation Loss: {final_val_loss:.4f}, Validation Accuracy: {final_val_acc:.4f}")
    if final_val_acc > best_val_acc:
        best_val_acc = final_val_acc
        with open("best_model.pkl", "wb") as f:
            pickle.dump(nn, f)  # Save the entire model object
        wandb.save("best_model.pkl")  # Save to W&B cloud storage
        print("Best model updated!")


wandb.agent(sweep_id, function=train_sweep, count=5)
