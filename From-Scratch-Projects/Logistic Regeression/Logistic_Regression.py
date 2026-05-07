import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def train(self, X, Y):
        n_samples, m_features = X.shape
        self.weights = np.zeros(m_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (predictions - Y))
            db = (1/n_samples) * np.sum(predictions - Y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_predictions = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_predictions
    
    def accuracy(self, y_pred, y_test):
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        return accuracy
