#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


def get_architecture(input_size):
    architectures = [
        [input_size, 4, 1],
        [input_size, 8, 1],
        [input_size, 4, 4, 1],
        [input_size, 8, 4, 1],
        [input_size, 8, 8, 1],
        [input_size, 16, 8, 1]
    ]
    return architectures


# In[3]:


lambdas = [0.001, 0.01, 0.1]


# In[4]:


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, reg_lambda=0.01):
        np.random.seed(42)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        L = len(self.layer_sizes) - 1
        for l in range(1, L + 1):
            epsilon = np.sqrt(6) / np.sqrt(self.layer_sizes[l] + self.layer_sizes[l-1])
            parameters[f"W{l}"] = np.random.uniform(-epsilon, epsilon, (self.layer_sizes[l], self.layer_sizes[l-1]))
            parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))
        return parameters


# In[5]:


def calculate_sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    A = calculate_sigmoid(Z)
    return A * (1 - A)


# In[6]:


def forward_prop(model, X):
    L = len(model.layer_sizes) - 1
    activation_cache = {"A0": X}
    for l in range(1, L + 1):
        W, b = model.parameters[f"W{l}"], model.parameters[f"b{l}"]
        A_prev = activation_cache[f"A{l-1}"]
        Z = np.dot(W, A_prev) + b
        A = calculate_sigmoid(Z)
        activation_cache[f"Z{l}"] = Z
        activation_cache[f"A{l}"] = A
    return activation_cache

def calculate_cost(model, Y_hat, Y):
    m = Y.shape[1]
    cross_entropy_cost = -np.sum(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8)) / m
    L2_cost = sum(np.sum(np.square(model.parameters[f"W{l}"])) for l in range(1, len(model.layer_sizes)))
    reg_cost = (model.reg_lambda / (2 * m)) * L2_cost
    return cross_entropy_cost + reg_cost

def backprop(model, activation_cache, X, Y):
    grads = {}
    m = X.shape[1]
    L = len(model.layer_sizes) - 1
    Y_hat = activation_cache[f"A{L}"]

    for l in reversed(range(1, L + 1)):
        A_prev = activation_cache[f"A{l-1}"]
        Z_curr = activation_cache[f"Z{l}"]
        A_curr = activation_cache[f"A{l}"]
        if l == L:
            dZ = A_curr - Y
        else:
            dZ = dA * sigmoid_derivative(Z_curr)

        grads[f"dW{l}"] = (1/m) * np.dot(dZ, A_prev.T) + (model.reg_lambda / m) * model.parameters[f"W{l}"]
        grads[f"db{l}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA = np.dot(model.parameters[f"W{l}"].T, dZ)

    return grads

def update(model, grads):
    L = len(model.layer_sizes) - 1
    for l in range(1, L + 1):
        model.parameters[f"W{l}"] -= model.learning_rate * grads[f"dW{l}"]
        model.parameters[f"b{l}"] -= model.learning_rate * grads[f"db{l}"]

def train_model(model, X, Y, num_iterations=1000, print_cost=False):
    for i in range(num_iterations):
        activation_cache = forward_prop(model, X)
        cost = calculate_cost(model, activation_cache[f"A{len(model.layer_sizes)-1}"], Y)
        grads = backprop(model, activation_cache, X, Y)
        update(model, grads)
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")

def predict(model, X):
    activation_cache = forward_prop(model, X)
    A_final = activation_cache[f"A{len(model.layer_sizes)-1}"]
    return (A_final > 0.5).astype(int)


# In[7]:


def z_score_normalisation(X):
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def load_dataset(path, label_column, one_hot=False):
    data = pd.read_csv(path)

    if data[label_column].dtype == 'object':
        data[label_column] = data[label_column].astype('category').cat.codes

    y_raw = data[label_column].values.reshape(-1, 1)
    X_raw = data.drop(columns=[label_column])

    if one_hot:
        encoder_y = OneHotEncoder(sparse_output=False)
        y = encoder_y.fit_transform(y_raw).T
    else:
        y = y_raw.T

    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    encoder_X = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = encoder_X.fit_transform(X_raw[categorical_cols]) if categorical_cols else np.empty((len(X_raw), 0))
    X_num = X_raw[numeric_cols].to_numpy()

    X_combined = np.hstack([X_num, X_cat])
    X_combined = np.nan_to_num(X_combined)

    return X_combined.T.astype(np.float64), y


# In[8]:


def stratified_k_fold_split(X, y, k=5, seed=42):
    np.random.seed(seed)
    labels = np.unique(y)
    folds = [[] for _ in range(k)]

    for label in labels:
        label_indices = np.where(y[0] == label)[0]
        np.random.shuffle(label_indices)
        split_indices = np.array_split(label_indices, k)
        for i in range(k):
            folds[i].extend(split_indices[i])

    return folds

def evaluate_model(model_class, X, y, num_epochs=1000, learning_rate=0.01, reg_lambda=0.01, architecture=[30, 10, 1]):
    accs = []
    f1s = []

    folds = stratified_k_fold_split(X, y, k=5)
    k = len(folds)

    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]

        X_train, y_train = X[:, train_idx], y[:, train_idx]
        X_test, y_test = X[:, test_idx], y[:, test_idx]

        model = model_class(architecture, learning_rate=learning_rate, reg_lambda=reg_lambda)
        train_model(model, X_train, y_train, num_iterations=num_epochs)
        preds = predict(model, X_test)

        acc = np.mean(preds == y_test)
        f1 = f1_score(preds, y_test)

        accs.append(acc)
        f1s.append(f1)

    return accs, f1s


# In[9]:


def f1_score(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


# In[10]:


def run_experiments(dataset_path, label_column, dataset_name):
    print(f"\n=== Running experiments for: {dataset_name} ===\n")

    X, y = load_dataset(dataset_path, label_column)
    X = z_score_normalisation(X)

    input_size = X.shape[0]
    print(f"Number of input features after encoding: {input_size}\n")

    architectures = get_architecture(input_size)
    results = []

    for m_arch in architectures:
        for reg in lambdas:
            accs, f1s = evaluate_model(
                NeuralNetwork,
                X, y,
                num_epochs=1000,
                learning_rate=0.1,
                reg_lambda=reg,
                architecture=m_arch
            )
            results.append({
                "Architecture": str(m_arch),
                "Lambda": reg,
                "Accuracy": round(np.mean(accs), 4),
                "F1 Score": round(np.mean(f1s), 4)
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df


# In[11]:


if __name__ == "__main__":
    run_experiments(
        dataset_path="/Users/saumyapandey/MLHW4/supporting files/datasets/wdbc.csv",
        label_column="label",  # update this to the actual column name
        dataset_name="Breast Cancer"
    )
    run_experiments(
        dataset_path="/Users/saumyapandey/MLHW4/supporting files/datasets/loan.csv",
        label_column="label",  # update this to the actual column name
        dataset_name="Loan Eligibility"
    )
    run_experiments(
        dataset_path="/Users/saumyapandey/MLHW4/supporting files/datasets/raisin.csv",
        label_column="label",  # update this to the actual column name
        dataset_name="Raisin"
    )
    run_experiments(
        dataset_path="/Users/saumyapandey/MLHW4/supporting files/datasets/titanic.csv",
        label_column="label",  # update this to the actual column name
        dataset_name="titanic"
    )


# In[12]:


def compute_learning_curve(X, y, architecture, reg_lambda=0.01, step=10):

    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y.T, test_size=0.2, stratify=y.T, random_state=42
    )
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T

    m = X_train.shape[1]
    points = list(range(step, m+1, step))
    test_losses = []

    for num_train in points:
        model = NeuralNetwork(architecture, learning_rate=0.1, reg_lambda=reg_lambda)
        train_model(model, X_train[:, :num_train], y_train[:, :num_train], num_iterations=500)
        activation_cache = forward_prop(model, X_test)
        A_test = activation_cache[f"A{len(architecture)-1}"]
        test_cost = calculate_cost(model, A_test, y_test)
        test_losses.append(test_cost)

    return points, test_losses

def plot_learning_curve(points, losses, dataset_name):
    plt.figure(figsize=(8, 5))
    plt.plot(points, losses, marker='o')
    plt.title(f"Learning Curve – {dataset_name}")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Test Set Cost J")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[21]:


if __name__ == "__main__":

    X_wdbc, y_wdbc = load_dataset("/Users/saumyapandey/Machine-Learning-Models/datasets/wdbc.csv", label_column="label")
    X_wdbc = z_score_normalisation(X_wdbc)

    architecture_wdbc = [X_wdbc.shape[0], 4, 1]
    points, losses = compute_learning_curve(X_wdbc, y_wdbc, architecture=architecture_wdbc, reg_lambda=0.01, step=10)
    plot_learning_curve(points, losses, "WDBC")

    X_loan, y_loan = load_dataset("/Users/saumyapandey/Machine-Learning-Models/datasets/loan.csv", label_column="label")
    X_loan = z_score_normalisation(X_loan)

    architecture = [X_loan.shape[0], 8, 1]
    points, losses = compute_learning_curve(X_loan, y_loan, architecture=architecture, reg_lambda=0.01, step=10)
    plot_learning_curve(points, losses, "Loan")

    X_raisin, y_raisin = load_dataset("/Users/saumyapandey/Machine-Learning-Models/datasets/raisin.csv", label_column="label")
    X_raisin = z_score_normalisation(X_raisin)

    architecture = [X_raisin.shape[0], 8, 1]
    points, losses = compute_learning_curve(X_raisin, y_raisin, architecture=architecture, reg_lambda=0.01, step=10)
    plot_learning_curve(points, losses, "Raisin")

    X_titanic, y_titanic = load_dataset("/Users/saumyapandey/Machine-Learning-Models/datasets/titanic.csv", label_column="label")
    X_titanic = z_score_normalisation(X_titanic)

    architecture = [X_titanic.shape[0], 8, 1]
    points, losses = compute_learning_curve(X_titanic, y_titanic, architecture=architecture, reg_lambda=0.01, step=10)
    plot_learning_curve(points, losses, "titanic")


# In[ ]:


def sigmoid_derivative(a):
    return a * (1 - a)


# In[22]:


def verify_backprop_example1():
    X = np.array([[0.13, 0.42]])
    Y = np.array([[0.90, 0.23]])
    m = X.shape[1]

    model = NeuralNetwork(layer_sizes=[1, 2, 1], learning_rate=0.1, reg_lambda=0.0)
    model.parameters['W1'] = np.array([[0.1], [0.2]])
    model.parameters['b1'] = np.array([[0.4], [0.3]])
    model.parameters['W2'] = np.array([[0.5, 0.6]])
    model.parameters['b2'] = np.array([[0.7]])

    print("\n--------------------------------------------")
    print("Computing the error/cost, J, of the network")

    total_cost = 0
    grads_sum = {"dW1": np.zeros((2, 1)), "db1": np.zeros((2, 1)), "dW2": np.zeros((1, 2)), "db2": np.zeros((1, 1))}

    for i in range(m):
        xi = X[:, i:i+1]
        yi = Y[:, i:i+1]

        activation_cache = forward_prop(model, xi)
        a2 = activation_cache['A1']
        a3 = activation_cache['A2']
        z2 = activation_cache['Z1']
        z3 = activation_cache['Z2']

        print(f"\tProcessing training instance {i+1}")
        print(f"\tForward propagating the input [{xi[0,0]:.5f}]")
        print(f"\t\ta1: [1.00000   {xi[0,0]:.5f}]")

        print(f"\n\t\tz2: [{z2[0,0]:.5f}   {z2[1,0]:.5f}]")
        print(f"\t\ta2: [1.00000   {a2[0,0]:.5f}   {a2[1,0]:.5f}]")

        print(f"\n\t\tz3: [{z3[0,0]:.5f}]")
        print(f"\t\ta3: [{a3[0,0]:.5f}]")

        cost = - (yi * np.log(a3 + 1e-8) + (1 - yi) * np.log(1 - a3 + 1e-8))
        print(f"\n\t\tf(x): [{a3[0,0]:.5f}]")
        print(f"\tPredicted output for instance {i+1}: [{a3[0,0]:.5f}]")
        print(f"\tExpected output for instance {i+1}: [{yi[0,0]:.5f}]")
        print(f"\tCost, J, associated with instance {i+1}: {cost[0,0]:.3f}\n")
        total_cost += cost[0,0]

        grads = backprop(model, activation_cache, xi, yi)
        for k in grads_sum:
            grads_sum[k] += grads[k]

        print(f"\tComputing gradients based on training instance {i+1}")
        delta3 = a3 - yi
        delta2 = (model.parameters['W2'].T @ delta3) * sigmoid_derivative(a2)
        print(f"\t\tdelta3: [{delta3[0,0]:.5f}]")
        print(f"\t\tdelta2: [{delta2[0,0]:.5f}   {delta2[1,0]:.5f}]\n")

        print(f"\t\tGradients of Theta2 based on training instance {i+1}:")
        print(f"\t\t\t{grads['db2'][0,0]:.5f}  {grads['dW2'][0,0]:.5f}  {grads['dW2'][0,1]:.5f}  \n")
        print(f"\t\tGradients of Theta1 based on training instance {i+1}:")
        print(f"\t\t\t{grads['db1'][0,0]:.5f}  {grads['dW1'][0,0]:.5f}  ")
        print(f"\t\t\t{grads['db1'][1,0]:.5f}  {grads['dW1'][1,0]:.5f}  \n")

    print(f"Final (regularized) cost, J, based on the complete training set: {total_cost/m:.5f}\n")

    print("\n\n--------------------------------------------")
    print("Running backpropagation")
    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    print("\n\t\tFinal regularized gradients of Theta1:")
    print(f"\t\t\t{(grads_sum['db1'][0,0]/m):.5f}  {(grads_sum['dW1'][0,0]/m):.5f}  ")
    print(f"\t\t\t{(grads_sum['db1'][1,0]/m):.5f}  {(grads_sum['dW1'][1,0]/m):.5f}  ")

    print("\n\t\tFinal regularized gradients of Theta2:")
    print(f"\t\t\t{(grads_sum['db2'][0,0]/m):.5f}  {(grads_sum['dW2'][0,0]/m):.5f}  {(grads_sum['dW2'][0,1]/m):.5f}  ")


if __name__ == "__main__":
    verify_backprop_example1()


# In[23]:


def verify_backprop_example2():
    lambd = 0.25
    print(f"Regularization parameter lambda={lambd:.3f}\n")
    print("Initializing the network with the following structure (number of neurons per layer): [2 4 3 2]\n")

    Theta1 = np.array([
        [0.150, 0.400],
        [0.100, 0.540],
        [0.190, 0.420],
        [0.350, 0.680]
    ])
    b1 = np.array([[0.420], [0.720], [0.010], [0.300]])

    Theta2 = np.array([
        [0.670, 0.140, 0.960, 0.870],
        [0.420, 0.200, 0.320, 0.890],
        [0.560, 0.800, 0.690, 0.090]
    ])
    b2 = np.array([[0.210], [0.870], [0.030]])

    Theta3 = np.array([
        [0.870, 0.420, 0.530],
        [0.100, 0.950, 0.690]
    ])
    b3 = np.array([[0.040], [0.170]])

    X = np.array([[0.320, 0.830], [0.680, 0.020]])
    Y = np.array([[0.750, 0.750], [0.980, 0.280]])

    model = NeuralNetwork([2, 4, 3, 2], learning_rate=0.1, reg_lambda=lambd)
    model.parameters['W1'] = Theta1
    model.parameters['b1'] = b1
    model.parameters['W2'] = Theta2
    model.parameters['b2'] = b2
    model.parameters['W3'] = Theta3
    model.parameters['b3'] = b3

    print("\n--------------------------------------------")
    print("Computing the error/cost, J, of the network")
    activation_cache = forward_prop(model, X)
    A1, A2, A3 = activation_cache['A1'], activation_cache['A2'], activation_cache['A3']

    X_size = X.shape[1]

    for i in range(X_size):
        print(f"\tProcessing training instance {i+1}")
        print(f"\tForward propagating the input [{X[0,i]:.5f}   {X[1,i]:.5f}]")
        print(f"\t\ta1: [1.00000   {X[0,i]:.5f}   {X[1,i]:.5f}]")

        print(f"\n\t\ta2: [" + "  ".join(f"{val:.5f}" for val in A1[:,i]) + "]")
        print(f"\n\t\ta3: [" + "  ".join(f"{val:.5f}" for val in A2[:,i]) + "]")
        print(f"\n\t\ta4: [" + "  ".join(f"{val:.5f}" for val in A3[:,i]) + "]")

        print(f"\n\t\tf(x): [" + "  ".join(f"{val:.5f}" for val in A3[:,i]) + "]")
        print(f"\tPredicted output: [" + "  ".join(f"{val:.5f}" for val in A3[:,i]) + "]")
        print(f"\tExpected output:  [" + "  ".join(f"{val:.5f}" for val in Y[:,i]) + "]")
        cost_i = -np.sum(Y[:,i:i+1] * np.log(A3[:,i:i+1]) + (1 - Y[:,i:i+1]) * np.log(1 - A3[:,i:i+1]))
        print(f"\tCost, J, associated with instance {i+1}: {cost_i:.3f}\n")

    print("\n--------------------------------------------")
    print("Running backpropagation")

    grads = backprop(model, activation_cache, X, Y)

    for i in range(X_size):
        print(f"\tComputing gradients based on training instance {i+1}")
        delta4 = A3[:, i:i+1] - Y[:, i:i+1]
        delta3 = (model.parameters['W3'].T @ delta4) * sigmoid_derivative(A2[:, i:i+1])
        delta2 = (model.parameters['W2'].T @ delta3) * sigmoid_derivative(A1[:, i:i+1])

        print(f"\t\tdelta4: [" + "   ".join(f"{val[0]:.5f}" for val in delta4) + "]")
        print(f"\t\tdelta3: [" + "   ".join(f"{val[0]:.5f}" for val in delta3) + "]")
        print(f"\t\tdelta2: [" + "   ".join(f"{val[0]:.5f}" for val in delta2) + "]\n")

    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    for l in range(1, 4):
        dW = grads[f'dW{l}']
        db = grads[f'db{l}']
        print(f"\n\t\tFinal regularized gradients of Theta{l}:")
        for i in range(dW.shape[0]):
            row = [db[i, 0]] + list(dW[i])
            print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))

if __name__ == "__main__":
    verify_backprop_example2()


# In[ ]:




