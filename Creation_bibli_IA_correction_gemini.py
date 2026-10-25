# ==============================================================================
# WHAT WE ARE COVERING IN THE PROJECT:
# 1: Create the database, choose our data, and set up our problem/splits.
# 2: Build the model architecture (layers, weights initialization, activations).
# 3: Implement forward pass, loss functions, backpropagation, and the optimizer.
# 4: Train the model, tune parameters, evaluate, and plot results.
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np

# 1. CREATE DATASET (Using NumPy instead of PyTorch)
# Generate synthetic linear data: Y = weight * X + bias
np.random.seed(42)
X = np.arange(0, 1, 0.01) # Shape: (100,)

weight = 0.7
bias = 0.3
Y = bias + X * weight

print("X first 10 values:", X[:10])
print("Y first 10 values:", Y[:10])

# Split data into training (80%) and testing (20%) sets
split = 80
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]
print(f"Dataset splits - Train: {len(X_train)}, Test: {len(X_test)}")

# Function to visualize data and predictions
def plot_predictions(training_data=X_train,
                     training_labels=Y_train,
                     test_data=X_test,
                     test_labels=Y_test,
                     predictions=None):

    plt.figure(figsize=(10, 7))
    plt.scatter(training_data, training_labels, c="b", s=4, label="training data")
    plt.scatter(test_data, test_labels, c="y", s=4, label="test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="predictions")

    plt.legend(prop={"size": 14})
    plt.show()

# Plot initial dataset
plot_predictions()


# 2. BUILD MODEL (Manual implementation using NumPy - replacing PyTorch nn.Module)
class LinearRegressionModelNumPy:
    def __init__(self):
        # Initialize weights and bias randomly (reproducible with seed)
        # Note: For the final project, you will need multiple layers and initialization methods (e.g., He, Xavier)
        self.weight = np.random.randn()
        self.bias = np.random.randn()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Forward pass: linear equation y = w * x + b
        return self.weight * x + self.bias

# Initialize our custom NumPy model
np.random.seed(42)
model_0 = LinearRegressionModelNumPy()

print("Initial parameters:", {"weight": model_0.weight, "bias": model_0.bias})

# Make initial predictions on test data (without updating parameters)
Y_prev = model_0.forward(X_test)
print("Initial predictions on test data:", Y_prev)

# Visualize initial random predictions against test data
plot_predictions(predictions=Y_prev)


# 3. LOSS FUNCTION AND OPTIMIZER (Manual implementation required for the project)
# Mean Absolute Error (MAE) loss function (equivalent to torch.nn.L1Loss)
def loss_fn(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

# Hyperparameters
learning_rate = 0.02
epochs = 100

# Track metrics over time
list_weight = []
list_bias = []
list_loss = []
list_loss_test = []


# 4. TRAINING LOOP (Manual Forward, Loss, Backpropagation, and Gradient Descent)
for epoch in range(epochs):
    
    # --- FORWARD PASS (Training data) ---
    Y_pred = model_0.forward(X_train)
    
    # --- CALCULATE LOSS ---
    loss = loss_fn(Y_pred, Y_train)
    
    # --- BACKPROPAGATION & GRADIENT CALCULATION ---
    # (For a simple linear model, derivatives of MAE/MSE can be computed analytically. 
    # For your MLP project, you will use the chain rule across layers).
    # Here using MSE derivative example or numerical approximation for gradients:
    # d_weight = mean(2 * (Y_pred - Y_train) * X_train) -> adjusted for demonstration
    error = Y_pred - Y_train
    d_weight = np.mean(2 * error * X_train)
    d_bias = np.mean(2 * error)
    
    # --- OPTIMIZER STEP (Gradient Descent update) ---
    model_0.weight -= learning_rate * d_weight
    model_0.bias -= learning_rate * d_bias

    # --- EVALUATION ON TEST DATA ---
    test_pred = model_0.forward(X_test)
    test_loss = loss_fn(test_pred, Y_test)

    # Save history
    list_weight.append(model_0.weight)
    list_bias.append(model_0.bias)
    list_loss.append(loss)
    list_loss_test.append(test_loss)

    # Print progress every 30 epochs
    if epoch % 30 == 0:
        print(f"Loss (Train): {loss:.4f} | Loss (Test): {test_loss:.4f} | Epoch: {epoch}")
        print(f"Parameters -> weight: {model_0.weight:.4f}, bias: {model_0.bias:.4f}\n")


# 5. PLOT TRAINING RESULTS
plt.figure()
plt.plot(list_weight)
plt.title("Weight adjustment over epochs")
plt.xlabel("Epoch")
plt.ylabel("Weight value")

plt.figure()
plt.plot(list_bias)
plt.title("Bias adjustment over epochs")
plt.xlabel("Epoch")
plt.ylabel("Bias value")

plt.figure()
plt.plot(list_loss, label="train loss")
plt.plot(list_loss_test, label="test loss")
plt.title("Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()