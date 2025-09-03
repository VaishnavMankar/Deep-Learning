""" Import Required Libraries """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

"""  Load Data """
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)   

""" Normalize Features """
X = (X - X.mean(axis=0)) / X.std(axis=0)

""" One-Hot Encode Labels """
encoder = OneHotEncoder(sparse_output=False)
y_encode = encoder.fit_transform(y)

n_samples, n_features = X.shape
n_classes = y_encode.shape[1]

""" Initialize Parameters """
np.random.seed(42)
W = np.random.randn(n_features, n_classes) * 0.01  
b = np.zeros((1, n_classes))                       

""" Softmax Function """
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

""" Cross-Entropy Loss """
def cross_entropy(y_true, y_pred):
    eps = 1e-9 
    m = y_true.shape[0]
    log_preds = np.log(y_pred + eps)          
    loss = -np.sum(y_true * log_preds) / m
    return loss

""" Forward Pass """
def forward(X, W, b):
    logits = np.dot(X, W) + b
    y_pred = softmax(logits)
    return y_pred

""" Backward Pass """
def backward(X, y_true, y_pred):
    n = X.shape[0]
    dW = np.dot(X.T, (y_pred - y_true)) / n
    db = np.sum(y_pred - y_true, axis=0, keepdims=True) / n
    return dW, db

""" Training Loop """
lr = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    # forward
    y_pred = forward(X, W, b)
    # loss
    loss = cross_entropy(y_encode, y_pred)
    losses.append(loss)
    # backward
    dW, db = backward(X, y_encode, y_pred)
    # update
    W -= lr * dW
    b -= lr * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")


y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = y.flatten()

accuracy = accuracy_score(y_true_labels, y_pred_labels)
print("\n")
print("Accuracy:", accuracy)
print("\n")

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print("\n")
print("Confusion Matrix:", conf_matrix)
print("\n")

""" Loss  curve plot"""
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()