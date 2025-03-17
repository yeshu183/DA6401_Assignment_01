class NN:
  def __init__(self,input_shape,output_shape,n_hidden_layers,h_per_layer,optimizer,activation_func="relu",loss_func="cross_entropy_loss",init_type="random",l2_reg=0):
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.n_h = n_hidden_layers
    self.k = h_per_layer
    self.weights,self.biases = self.weight_init(init_type)
    self.optimizer_function = optimizer
    self.grad_weights = [0]*(self.n_h+1)
    self.grad_biases = [0]*(self.n_h+1)
    self.activation_func = activation_func
    self.loss_func = loss_func
    self.l2_reg = l2_reg
  def activation(self,x):
    if self.activation_func == "relu":
      return np.maximum(0,x)
    elif self.activation_func == "tanh":
      return np.tanh(x)
    elif self.activation_func == "sigmoid":
      return 1/(1+np.exp(-x))
  def activation_grad(self,x):
    if self.activation_func == "relu":
      return np.where(x>0,1,0)
    elif self.activation_func == "tanh":
      return 1-np.tanh(x)**2
    elif self.activation_func == "sigmoid":
      s = 1/(1+np.exp(-x))
      return s*(1-s)
  def softmax(self,x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
  def one_hot(self,y):
    ohe = np.zeros((len(y), self.output_shape))
    for i in range(len(y)):
      ohe[i][y[i]] = 1
    return ohe
  def loss(self,y_hat,y_true):
    if self.loss_func == "cross_entropy_loss":
      y_hat = np.array(y_hat)
      y_true = np.array(y_true)
      epsilon = 1e-10  # Prevent log(0)
      return -np.sum(y_true * np.log(y_hat + epsilon)) / y_hat.shape[0]
    elif self.loss_func == "squared_error":
      return np.sum((y_hat-y_true)**2)
  def accuracy(self,y_hat,y_true):
    return np.sum(y_hat==y_true)/len(y_true)
  def weight_init(self, init_type="random"):
    weights = []
    biases = []
    if init_type == "random":
        weights.append(np.random.randn(self.input_shape, self.k))
        biases.append(np.random.randn(self.k, 1))
        for i in range(self.n_h - 1):
            weights.append(np.random.randn(self.k, self.k))
            biases.append(np.random.randn(self.k, 1))
        weights.append(np.random.randn(self.k, self.output_shape))
        biases.append(np.random.randn(self.output_shape, 1))
    elif init_type == "xavier":
        # Xavier Initialization
        weights.append(np.random.randn(self.input_shape, self.k) * np.sqrt(2 / (self.input_shape + self.k)))
        biases.append(np.zeros((self.k, 1)))  # Biases are usually initialized to 0
        for i in range(self.n_h - 1):
            weights.append(np.random.randn(self.k, self.k) * np.sqrt(2 / (self.k + self.k)))
            biases.append(np.zeros((self.k, 1)))
        weights.append(np.random.randn(self.k, self.output_shape) * np.sqrt(2 / (self.k + self.output_shape)))
        biases.append(np.zeros((self.output_shape, 1)))
    return weights, biases

  def update_parameters(self):
    for i in range(len(self.weights)):
      grad_w = np.array(self.grad_weights[i])
      grad_b = np.array(self.grad_biases[i])
      self.weights[i], self.biases[i] = self.optimizer_function.step(
              self.weights[i], self.biases[i],grad_w,grad_b,key=f"layer_{i}")

  def forward(self,x):
    a_list = [0]*(self.n_h+2)
    h_list = [0]*(self.n_h+2)
    h = x.T
    h_list[0] = h
    for i in range(self.n_h):
      a_list[i+1] = np.dot(self.weights[i].T,h_list[i])+self.biases[i] #biases are broadcasted
      h_list[i+1] = self.activation(a_list[i+1])
    a_list[self.n_h+1] = np.dot(self.weights[self.n_h].T,h_list[self.n_h])+self.biases[self.n_h]
    y_hat = self.softmax(a_list[self.n_h+1]).T
    h_list[self.n_h+1] = y_hat
    return a_list,h_list
  def backward(self,a_list,h_list,y):
    a_grad_list = [0]*(self.n_h+2)
    h_grad_list = [0]*(self.n_h+2)
    y_hat = h_list[self.n_h+1]
    if self.loss_func == "cross_entropy_loss": #gradient wrt output layer
      a_grad = (y_hat - self.one_hot(y)).T
    elif self.loss_func == "squared_error":
      a_grad = 2*(np.argmax(y_hat,axis=0) - y) #must be changed appropriately
    a_grad_list[-1] = a_grad
    for k in range(self.n_h,-1,-1): # gradient wrt hiddden layers
      h_grad = np.dot(self.weights[k],a_grad_list[k+1])
      h_grad_list[k] = h_grad
      a_grad = np.multiply(h_grad,self.activation_grad(a_list[k]))
      a_grad_list[k] = a_grad
      self.grad_weights[k] = np.dot(a_grad_list[k+1],h_list[k].T).T + + self.l2_reg * self.weights[k] #gradients wrt parameters
      self.grad_biases[k] = np.sum(a_grad_list[k+1],axis=1, keepdims=True)
  def predict(self,x):
    a_list,h_list = self.forward(x)
    return np.argmax(h_list[self.n_h+1],axis=0)
  def train(self,x_train,y_train,x_val,y_val,epochs=10,batch_size=32):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for i in range(epochs):
      start = time.time()
      indices = np.arange(len(x_train))
      np.random.shuffle(indices)
      # if self.opt_name == 'sgd':
      #   batch_size=1
      # else:
      #   batch_size = batch_size
      per_epoch_loss = []
      per_epoch_acc = []
      for j in range(0,len(x_train),batch_size):
        #startf = time.time()
        batch_indices = indices[j:j+batch_size]
        x_batch = x_train[batch_indices]
        y_batch = y_train[batch_indices]
        a_list_train,h_list_train = self.forward(x_batch) # forward pass
        #end_time_f = time.time()
        #startb = time.time()
        #print("forward pass done")
        self.backward(a_list_train,h_list_train,y_batch) # backward pass
        #end_time_b = time.time()
        #starto = time.time()
        #print("backward pass done")
        self.update_parameters()
        #self.optimizer_function(lr=lr) # updating weights
        #end_time_o = time.time()
        #print("optimizer done")
        y_hat = h_list_train[self.n_h+1]
        per_epoch_loss.append(self.loss(y_hat,self.one_hot(y_batch)))
        per_epoch_acc.append(self.accuracy(np.argmax(np.array(y_hat),axis=1),y_batch))
      train_loss = np.mean(per_epoch_loss)
      train_acc = np.mean(per_epoch_acc)
      train_loss_list.append(train_loss)
      train_acc_list.append(train_acc)
      #print(f"Epoch: {i+1} | Batch: {j+1}/{len(x_train)} | f_time: {(end_time_f-startf): .6f} | b_time:{(end_time_b-startb): .6f} | o_time:{(end_time_o-starto): .6f}")
      #print("Train Loss: ",train_loss,"Train Accuracy: ",train_acc)
      a_list_val,h_list_val = self.forward(x_val)
      y_hat_val = h_list_val[self.n_h+1]
      val_loss = self.loss(y_hat_val,self.one_hot(y_val))
      val_acc = self.accuracy(np.argmax(np.array(y_hat_val),axis=1),y_val)
      val_loss_list.append(val_loss)
      val_acc_list.append(val_acc)
      end = time.time()
      # Log metrics
      wandb.log({
          "train_loss": train_loss,
          "train_acc": train_acc,
          "val_loss": val_loss,
          "val_acc": val_acc,
          "epoch": i+1
      })
      print(f"Epoch: {i+1} | Time: {(end-start):.2f} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
    return train_loss_list,train_acc_list,val_loss_list,val_acc_list