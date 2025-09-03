import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and Prepare the data
iris = load_iris()
X = iris.data.astype(float)  
y = iris.target

# Normalize features
X = (X-X.mean(axis=0))/(X.std(axis=0))

# Convert labels to one-hot encoding
def one_hot(y,num_classes):
    Y = np.zeros((y.size,num_classes))
    Y[np.arange(y.size),y]=1
    return Y

#Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
Y_train = one_hot(y_train,3)
Y_test = one_hot(y_test,3)

#Implement Softmax Regression from Scratch
def softmax(z):
    z = z - np.max(z, axis=1,keepdims=True)
    exp_z = np.exp(z)
    output = exp_z / np.sum(exp_z,axis=1,keepdims=True)
    return output

def cross_entropy(y_true,y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred,eps,1-eps)
    ce = (-1) * np.mean(np.sum(y_true * np.log(y_pred),axis=1))
    return ce

#Train the Model
np.random.seed(0)
n_features = X_train.shape[1]
n_classes = 3
W = np.random.randn(n_features,n_classes) * 0.01  
b = np.zeros((1,n_classes))                   
lr = 0.1                                        
epochs = 500
loss_history = []

for epoch in range(1,epochs+1):
    # Forward pass
    logits = np.dot(X_train,W) + b
    probs = softmax(logits)
    loss = cross_entropy(Y_train,probs)
    loss_history.append(loss)

    # Backward pass
    N = X_train.shape[0]
    dlogits = (probs-Y_train) / N
    dW = np.dot(X_train.T,dlogits)
    db = np.sum(dlogits,axis=0, keepdims=True)

    # Parameter update
    W -= lr*dW
    b -= lr*db

    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

#Evaluate the Model
def predict(X):
    logits = np.dot(X,W) + b  
    probs = softmax(logits)
    return np.argmax(probs,axis=1)

y_pred_train = predict(X_train)
y_pred_test = predict(X_test)
train_accuracy = np.mean(y_pred_train==y_train)
test_accuracy = np.mean(y_pred_test==y_test)

print("\nFinal Results:")
print(f"Train accuracy: {train_accuracy*100:.2f}%")
print(f"Test  accuracy: {test_accuracy*100:.2f}%")

# Confusion matrix
def confusion_matrix(y_true,y_pred,n_classes):
    cm = np.zeros((n_classes,n_classes),dtype=int)
    for t,p in zip(y_true,y_pred):
        cm[t,p]+=1
    return cm

cm = confusion_matrix(y_test, y_pred_test, n_classes)
print("\nConfusion Matrix:\n", cm)

# Plot loss curve
plt.figure()
plt.plot(np.arange(1,epochs+1),loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Softmax Regression Loss (Train)")
plt.tight_layout()
plt.show()

# heatmap
plt.figure()
plt.imshow(cm,cmap='Blues')
plt.title('Confusion Matrix for test data')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(n_classes),iris.target_names)
plt.yticks(np.arange(n_classes),iris.target_names)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j,i,cm[i,j],ha='center',va='center')
plt.tight_layout()
plt.show()