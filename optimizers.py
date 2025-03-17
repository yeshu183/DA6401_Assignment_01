import numpy as np
class Optimizers:
    def __init__(self, lr=1e-3, epsilon=1e-8, beta1=0.9, beta2=0.999, rho=0.9, optimizer="adam"):
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.rho = rho
        self.optimizer = optimizer  # Store optimizer type
        
        self.v, self.s, self.m = {}, {}, {}  # Optimizer state
        self.t = 0  # Time step for Adam/Nadam

    def _initialize_params(self, key, shape):
        """Initialize optimizer states for a given parameter key"""
        if key not in self.v:
            self.v[key] = np.zeros(shape)
            self.s[key] = np.zeros(shape)
            self.m[key] = np.zeros(shape)

    def step(self, w, b, dw, db, key="w"):
        """Generic function to perform an optimization step based on selected optimizer"""
        self._initialize_params(key, w.shape)
        self.t += 1  # Increment time step
        
        if self.optimizer == "sgd":
            w -= self.lr * dw
            b -= self.lr * db
            
        elif self.optimizer == "momentum":
            self.v[key] = self.beta1 * self.v[key] - self.lr * dw
            w += self.v[key]
            b -= self.lr * db
            
        elif self.optimizer == "nesterov":
            prev_v = self.v[key]
            self.v[key] = self.beta1 * self.v[key] - self.lr * dw
            w += -self.beta1 * prev_v + (1 + self.beta1) * self.v[key]
            b -= self.lr * db

        elif self.optimizer == "rmsprop":
            self.s[key] = self.rho * self.s[key] + (1 - self.rho) * (dw ** 2)
            w -= self.lr * dw / (np.sqrt(self.s[key]) + self.epsilon)
            b -= self.lr * db

        elif self.optimizer == "adam":
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * dw
            self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (dw ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            s_hat = self.s[key] / (1 - self.beta2 ** self.t)
            w -= self.lr * m_hat / (np.sqrt(s_hat) + self.epsilon)
            b -= self.lr * db
        
        return w, b