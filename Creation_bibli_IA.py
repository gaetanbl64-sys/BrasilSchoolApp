#what_we_are_covering = 
#{
# 1: "Creer la base de donnée, choisir nos données et create our cas", 
# 2: "Build model, the way to adjust automaticly parameters to our cas base on our training data",
# 3: "Fitting the model to data (training)",
# 4: "..."
#}
#---------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0, 1, 0.01)
split = 80

X_train = X[:split]
X_test = X[split:]

# weight = 0.7
# bias = 0.3
# Y = bias + X * weight
# Y_train = Y[:split]
# Y_test = Y[split:]
# print(Y)

#---------------------------------------------
print(X[:10])
print("Lenght of echantillons")
print(len(X_train), len(X_test))
#---------------------------------------------


def plot_predictions(X, Y, test_data, test_labels):
    plt.figure(figsize=(10, 7))

    plt.scatter(X, Y, c="b", s=4, label = "training data")  # it's like plt.plot() but with cloud of point
    plt.scatter(test_data, test_labels, c="y", s=4, label = "test data")

    plt.legend(prop={"size":14});
    plt.show()

class LinearRegressionModel(): 
    np.random.seed(42)
    def __init__(self, numberLayers=1 , numberNeuronesPerLayers=1):
        self.numberLayers = numberLayers
        self.numberNeuronesPerLayers = numberNeuronesPerLayers
        self.init_weights = np.random.randn(numberLayers*numberNeuronesPerLayers)  # or np.random.randn(numberLayers, numberNeuronesPerLayers) ?
        self.init_bias = np.random.randn(numberLayers*numberNeuronesPerLayers)

    def forward(self, x):
        return self.init_weights * x + self.init_bias

model_0 = LinearRegressionModel()

Y_init_prevision = model_0.forward(X_train)
Y_real_test = model_0.forward(X_test)

#---------------------------------------------
print(model_0.init_weights, model_0.init_bias)
print(Y_init_prevision)
plot_predictions(X_train, Y_init_prevision, X_test, Y_real_test);
#---------------------------------------------

class LearningClass():
    def __init__(self, epochs = 100, stop_precision = 0.05, learning_rate = 0.02, frequency_save = 30):
        self.epochs = epochs
        self.stop_precision = stop_precision
        self.learning_rate = learning_rate
        self.frequency_save = frequency_save
        self.list_bias = []
        self.list_weight = []

    def learningLoop(self, epochs, frequency_save, list_weight, list_bias):
      # for epoch in range(epochs):

      #   # 1. Forward pass with only X_train
      #   Y_pred = model_0(X_train)
        
      #   # 2. Calculate the loss
      #   loss = loss_fn(Y_pred, Y_train)
        
      #   # 3. Optimizer zero grad
      #   optimizer.zero_grad()
        
      #   # 4. Perform backpropagation on the loss, calculate grad of each parameter
      #   loss.backward()
        
      #   # 5. Step the optimizer, perform gradient descent and adjust parameters
      #   optimizer.step() # by default, the optimizer accumulate through the loop so we have to zero them #3

          # if (epoch % self.frequency_save == 0):
          #     list_weight.append()
          #     list_bias.append()

      return list_weight[-1], list_bias[-1]

        
myFirstLearning = LearningClass()
[final_weight, final_bias] = myFirstLearning.learningLoop()

#---------------------------------------------
print(final_weight, final_bias)
#---------------------------------------------