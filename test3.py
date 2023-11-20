
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return exp_values / np.sum(exp_values)

X = np.random.rand(1, 756)  # Input data with shape (1, 756)
W1 = np.random.rand(756, 128)  # Weight matrix for the first hidden layer with shape (756, 128)
B1 = np.random.rand(1, 128)  # Bias vector for the first hidden layer with shape (1, 128)

print(W1.shape)  # (756, 128)
print(X.shape)  # (1, 756)
print(B1.shape)  # (1, 128)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(A1.shape)  # (1, 128)
print(Z1.shape)  # (1, 128)

W2 = np.random.rand(128, 64)  # Weight matrix for the second hidden layer with shape (128, 64)
B2 = np.random.rand(1, 64)  # Bias vector for the second hidden layer with shape (1, 64)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2.shape)  # (1, 64)
print(Z2.shape)  # (1, 64)

W3 = np.random.rand(64, 32)  # Weight matrix for the third hidden layer with shape (64, 32)
B3 = np.random.rand(1, 32)  # Bias vector for the third hidden layer with shape (1, 32)

A3 = np.dot(Z2, W3) + B3
Z3 = sigmoid(A3)
print(A3.shape)  # (1, 32)
print(Z3.shape)  # (1, 32)

W4 = np.random.rand(32, 10)  # Weight matrix for the output layer with shape (32, 10)
B4 = np.random.rand(1, 10)  # Bias vector for the output layer with shape (1, 10)

A4 = np.dot(Z3, W4) + B4
Y = softmax(A4)
print(A4.shape)  # (1, 10)
print(Y.shape)  # (1, 10)


---------------------

This code demonstrates a simple feedforward neural network with four hidden layers and an output layer. Here's a step-by-step explanation:

1. The code imports the NumPy library for numerical operations.

2. The `sigmoid` function is defined, which calculates the sigmoid activation function for a given input `x`. The sigmoid function returns values between 0 and 1, which are commonly used to model neuron activations.

3. The `softmax` function is defined, which calculates the softmax activation function for a given input `x`. The softmax function normalizes the input values into a probability distribution, commonly used for multi-class classification problems.

4. The code initializes the input data `X` with random values, creating an array of shape (1, 756). This means there is one sample with 756 features.

5. The weight matrix `W1` and bias vector `B1` are initialized for the first hidden layer. `W1` has a shape of (756, 128), indicating 756 input features and 128 hidden units, and `B1` has a shape of (1, 128), providing biases for each hidden unit.

6. The shapes of `W1`, `X`, and `B1` are printed, confirming the dimensions.

7. The activation `A1` of the first hidden layer is computed by multiplying `X` with `W1` and adding `B1`. The dot product of `X` and `W1` results in an array of shape (1, 128), and then `B1` is broadcasted and added to each row.

8. The sigmoid activation function is applied to `A1`, producing the hidden layer output `Z1` with the same shape as `A1`.

9. The shapes of `A1` and `Z1` are printed to confirm their dimensions.

10. The process is repeated for the subsequent layers: `W2` and `B2` are initialized for the second hidden layer, and the output of the first hidden layer (`Z1`) is multiplied by `W2`, added `B2`, passed through the sigmoid activation function, and stored in `Z2`. The shapes of `A2` and `Z2` are printed.

11. The same steps are repeated for the third hidden layer using `W3` and `B3`. The output of the second hidden layer (`Z2`) is multiplied by `W3`, added `B3`, passed through the sigmoid activation function, and stored in `Z3`. The shapes of `A3` and `Z3` are printed.

12. Finally, the output layer is constructed. `W4` and `B4` are initialized for the output layer. The output of the third hidden layer (`Z3`) is multiplied by `W4`, added `B4`, and stored in `A4`. Then, the softmax activation function is applied to `A4` to obtain the final output `Y`, which represents the predicted probabilities for each class. The shapes of `A4` and `Y` are printed.

In summary, this code sets up a neural network with four hidden layers and an output layer, applies sigmoid activation to the hidden layers and softmax activation to the output layer, and prints the shapes of the intermediate and final outputs.


-------------------------

**Neural Network Report**

This code implements a simple feedforward neural network with four hidden layers and an output layer. The neural network architecture consists of the following layers:

1. Input Layer: The input data has a shape of (1, 756), meaning there is one sample with 756 features.

2. First Hidden Layer: This layer has 128 hidden units. The weight matrix (`W1`) connecting the input layer to the first hidden layer has a shape of (756, 128), and the bias vector (`B1`) has a shape of (1, 128).

3. Second Hidden Layer: This layer has 64 hidden units. The weight matrix (`W2`) connecting the first hidden layer to the second hidden layer has a shape of (128, 64), and the bias vector (`B2`) has a shape of (1, 64).

4. Third Hidden Layer: This layer has 32 hidden units. The weight matrix (`W3`) connecting the second hidden layer to the third hidden layer has a shape of (64, 32), and the bias vector (`B3`) has a shape of (1, 32).

5. Output Layer: This layer produces the final output of the neural network. It has 10 units, corresponding to a multi-class classification problem. The weight matrix (`W4`) connecting the third hidden layer to the output layer has a shape of (32, 10), and the bias vector (`B4`) has a shape of (1, 10).

The activation function used in each hidden layer is the sigmoid function (`sigmoid(x) = 1 / (1 + exp(-x))`). The sigmoid function squashes the input values between 0 and 1, allowing the network to model non-linear relationships.

The activation function used in the output layer is the softmax function (`softmax(x)`). It normalizes the input values into a probability distribution, which is suitable for multi-class classification problems.

During the computation, the code uses the NumPy library for efficient numerical operations. It initializes the weight matrices (`W1`, `W2`, `W3`, `W4`) and bias vectors (`B1`, `B2`, `B3`, `B4`) with random values.

The code then performs the forward propagation step:

1. The input data (`X`) is passed through the first hidden layer by computing the dot product of `X` and `W1`, adding the bias vector `B1`, and storing the result in `A1`. The sigmoid activation function is then applied to `A1`, producing the output `Z1`.

2. The output of the first hidden layer (`Z1`) is passed through the second hidden layer in a similar manner. The dot product of `Z1` and `W2` is computed, the bias vector `B2` is added, and the result is stored in `A2`. The sigmoid activation function is applied to `A2`, resulting in `Z2`.

3. The process is repeated for the third hidden layer. The dot product of `Z2` and `W3` is computed, the bias vector `B3` is added, and the result is stored in `A3`. The sigmoid activation function is applied to `A3`, yielding `Z3`.

4. Finally, the output layer is computed. The dot product of `Z3` and `W4` is computed, the bias vector `B4` is added, and the result is stored in `A4`. The softmax activation function is applied to `A4`, producing the final output `Y`, which represents the predicted probabilities for each class.

Throughout the code, the shapes of the weight matrices, input data, bias vectors,

 and intermediate outputs are printed to confirm their dimensions.

This neural network architecture can be used for various tasks, such as classification problems with multiple classes. The weights and biases are randomly initialized, and the outputs (`Z1`, `Z2`, `Z3`, `Y`) represent the activations of the hidden layers and the final predicted probabilities, respectively.