import numpy as np
class Optimizers:
    def __init__(self, lr=1e-3, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon  # Small value to prevent division by zero

        # Initialize state dictionaries
        self.v = {}  # Velocity (for momentum, NAG)
        self.s = {}  # Squared gradients (for RMSprop, Adam, Nadam)
        self.m = {}  # First moment (for Adam/Nadam)
        self.t = 0   # Time step (for Adam/Nadam)

    def _initialize_params(self, w_key, param_shape):
        """Initialize optimizer states for a given parameter key"""
        if w_key not in self.v:
            self.v[w_key] = np.zeros(param_shape)
            self.s[w_key] = np.zeros(param_shape)
            self.m[w_key] = np.zeros(param_shape)

    def sgd(self, w, b, dw, db, key="w"):
        """Vanilla Stochastic Gradient Descent"""
        w -= self.lr * dw
        b -= self.lr * db
        return w, b

    def momentum(self, w, b, dw, db, key="w", beta1=0.9):
        """Momentum-based Gradient Descent"""
        self._initialize_params(key, w.shape)
        self.v[key] = beta1 * self.v[key] - self.lr * dw
        w += self.v[key]
        b -= self.lr * db
        return w, b

    def nesterov(self, w, b, dw, db, key="w", beta1=0.9):
        """Nesterov Accelerated Gradient (NAG)"""
        self._initialize_params(key, w.shape)
        prev_v = self.v[key]
        self.v[key] = beta1 * self.v[key] - self.lr * dw
        w += -beta1 * prev_v + (1 + beta1) * self.v[key]
        b -= self.lr * db
        return w, b

    def rmsprop(self, w, b, dw, db, key="w", rho=0.9):
        """RMSprop Optimization"""
        self._initialize_params(key, w.shape)
        self.s[key] = rho * self.s[key] + (1 - rho) * (dw ** 2)
        w -= self.lr * dw / (np.sqrt(self.s[key]) + self.epsilon)
        b -= self.lr * db
        return w, b

    def adam(self, w, b, dw, db, key="w", beta1=0.9, beta2=0.999):
        """Adaptive Moment Estimation (Adam)"""
        self._initialize_params(key, w.shape)
        self.t += 1  # Time step
        self.m[key] = beta1 * self.m[key] + (1 - beta1) * dw
        self.s[key] = beta2 * self.s[key] + (1 - beta2) * (dw ** 2)

        m_hat = self.m[key] / (1 - beta1 ** self.t)
        s_hat = self.s[key] / (1 - beta2 ** self.t)

        w -= self.lr * m_hat / (np.sqrt(s_hat) + self.epsilon)
        b -= self.lr * db
        return w, b
