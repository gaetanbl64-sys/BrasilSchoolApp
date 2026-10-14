what_we_are_covering = {1: "Creer la base de donnée, choisir nos données et create our cas",
                        2: "Build model, the way to adjust automaticly parameters to our cas base on our training data",
                        3: "Fitting the model to data (training)",
                        4: "..."}

import torch
import torch.nn as nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import torch

X = torch.arange(0, 1, 0.01)
#X = X.unsqueeze(dim=1)

weight = 0.7
bias = 0.3
Y = bias + X * weight

print(X[:10])
print(Y)

split = 80

X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]
print(len(X_train), len(Y_train), len(X_test), len(Y_test))


def plot_predictions(training_data=X_train,
                     training_labels=Y_train,
                     test_data=X_test,
                     test_labels=Y_test,
                     predictions=None):

    plt.figure(figsize=(10, 7))

    plt.scatter(training_data, training_labels, c="b", s=4, label = "training data")
    plt.scatter(test_data, test_labels, c="y", s=4, label = "test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label = "predictions")

    plt.legend(prop={"size":14});

plot_predictions()

class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherhits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust it to the ideal weight
                                                 requires_grad=True,
                                                 dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, # <- start with a random bias and try to adjust it to the ideal bias
                                              requires_grad=True,
                                              dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()

# print(list(model_0.parameters())) # si on ne rajoute pas list, on print un générateur (une adresse)
print(model_0.state_dict())

with torch.inference_mode():
  Y_prev = model_0(X_test)

print(weight, bias)
print(Y_prev)

plot_predictions(predictions = Y_prev);

# loss function mean of the difference
loss_fn = nn.L1Loss() # = torch.mean(torch.abs(Y_test-Y_prev))

# optimizer stochastique gradient descent
optimizer = torch.optim.SGD(params = model_0.parameters(),
                            lr = 0.02)
print(model_0.state_dict(),"\n")
epochs = 100 # nbr of loop
list_weight=[]
list_bias=[]
list_loss=[]
list_loss_test=[]

for epoch in range(epochs):
  ### Set the model to training mode
  model_0.train() # train mode in PyTorch sets all parameters that require gradients to grad
  # 1. Forward pass with only X_train
  Y_pred = model_0(X_train)
  # 2. Calculate the loss
  loss = loss_fn(Y_pred, Y_train)
  # 3. Optimizer zero grad
  optimizer.zero_grad()
  # 4. Perform backpropagation on the loss, calculate grad of each parameter
  loss.backward()
  # 5. Step the optimizer, perform gradient descent and adjust parameters
  optimizer.step() # by default, the optimizer accumulate through the loop so we have to zero them #3

  ### Set the model to testing mode (desactivate Dropout and BatchNorm for example)
  model_0.eval()
  list_weight.append(model_0.state_dict()['weights'].item())
  list_bias.append(model_0.state_dict()['bias'].item())
  list_loss.append(loss.item())
  with torch.inference_mode():
    test_prev = model_0(X_test) # Y_prev_new only for the last prediction
    test_loss = loss_fn(test_prev, Y_test)
    list_loss_test.append(test_loss.item())
  if epoch % 30 == 0:
    print(f"Loss of training data: {loss} | Loss of testing data: {test_loss} | Epoch: {epoch}")
    print(model_0.state_dict(),"\n")

    print(list_loss)
    
plt.figure()
plt.plot(list_weight);
plt.title("weight")
plt.figure()
plt.plot(list_bias);
plt.title("bias")
plt.figure()
plt.plot(list_loss);
plt.plot(list_loss_test);
plt.title("loss")
plt.legend(["loss", "test_loss"])