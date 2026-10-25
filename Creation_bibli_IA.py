#what_we_are_covering = {
# 1: "Creer la base de donnée, choisir nos données et create our cas", 
# 2: "Build model, the way to adjust automaticly parameters to our cas base on our training data",
# 3: "Fitting the model to data (training)",
# 4: "..."}
#---------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import math

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

    def init_forward(self, x):
        return self.init_weights * x + self.init_bias

    def forward(self, x, w, b):
        return w * x + b

model_0 = LinearRegressionModel()

Y_correct_prediction = model_0.init_forward(X_train)
Y_real_test = model_0.init_forward(X_test)

#---------------------------------------------
print(model_0.init_weights, model_0.init_bias)
print(Y_correct_prediction)
#plot_predictions(X_train, Y_correct_prediction, X_test, Y_real_test);
#---------------------------------------------

class LearningClass():
    def __init__(self, epochs = 100, stop_precision = 0.05, learning_rate = 0.02, frequency_print = 10):
        self.epochs = epochs
        self.stop_precision = stop_precision
        self.learning_rate = learning_rate
        self.frequency_print = frequency_print

        self.list_bias = [np.random.randn()]
        self.list_weight = [np.random.randn()]

    def loss_fn(self, tmp, real):   # available method = {"SMAE":SMAE(), "MSE":...}
        lossS = []
        for i, j in zip(tmp, real):     # /!\ zip
            lossS.append((i-j)**2)   
        return np.average(lossS)
    
    def backward_fn(self, tmp, real):
        # every method of backward is linked to the loss_fn method SMAE, MSE...

        # derivate partial loss by weight and bias, example with weight :
        # loss = (i-j)**2 =
        # ([list_bias[-1]+X_test*list_weight[-1]] - Y_correct_prediction)**2
        # so by derivant par / à weight :
        # 2 * X_test * ([list_bias[-1]+X_test*list_weight[-1]] - Y_correct_prediction)
        # = 2 * X_test * erreur

        error = tmp - real
        grad_weight = np.average(error * X_train) * 2
        grad_bias = np.average(error) * 2
        return grad_weight, grad_bias
    
    def optimizerStep(self, backward):
        new_w = self.list_weight[-1] - self.learning_rate*backward[0]    # available method = {"SGD":, "Adams":...}
        new_b = self.list_bias[-1] - self.learning_rate*backward[1]
        return new_w, new_b

    def learningLoop(self):
        for epoch in range(self.epochs):
            # 1. Forward pass with only X_train
            Y_tmp_predition = model_0.forward(X_train, self.list_weight[-1], self.list_bias[-1])
        
            # 2. Calculate the loss
            loss = self.loss_fn(Y_tmp_predition, Y_correct_prediction)
        
            # 3. Optimizer zero grad only with nn
        
            # 4. Perform backpropagation on the loss, calculate grad of each parameter
            backward = self.backward_fn(Y_tmp_predition, Y_correct_prediction)
        
            # 5. Step the optimizer, perform gradient descent and adjust parameters
            new_w, new_b = self.optimizerStep(backward)

            self.list_weight.append(new_w)
            self.list_bias.append(new_b)

            if (epoch % self.frequency_print == 0):
                print(f"Epoch {epoch}: loss = {loss}, weight = {self.list_weight[-1]}, bias = {self.list_bias[-1]}")
                Y_real_test = model_0.forward(X_test, self.list_weight[-1], self.list_bias[-1])
                plot_predictions(X_train, Y_correct_prediction, X_test, Y_real_test)

        
myFirstLearning = LearningClass()
myFirstLearning.learningLoop()
