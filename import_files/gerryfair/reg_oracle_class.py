import numpy as np
from scipy.special import softmax

class RegOracle:
    """Class RegOracle, linear threshold classifier."""
    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b1

    def predict(self, X):
        """Predict labels on data set X."""
        reg0 = self.b0
        reg1 = self.b1
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            x_i = x_i.values.reshape(1, -1)
            c_0 = reg0.predict(x_i)
            c_1 = reg1.predict(x_i)
            #print(c_0, c_1)
            #sm = softmax(np.array([c_0, c_1]))
            #p = np.random.uniform()
            #y_i = int(p>sm[0]) 
            y_i = int(c_1 < c_0)
            
            y.append(y_i)
        return y
    
    def state_dict(self):
        return {'b0': self.b0.state_dict(), 'b1': self.b1.state_dict()}

class RandomLinearThresh:
    """Class random hyperplane classifier."""
    def __init__(self, d):
        self.coefficient = [np.random.uniform(-1, 1) for _ in range(d)]

    def predict(self, X):
        """Predict labels on data set X."""
        beta = self.coefficient
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            c_1 = np.dot(beta, x_i)
            
            #sm = softmax(np.array([0, c_1]))
            #p = np.random.uniform()
            #y_i = int(p>sm[0]) 
            y_i = int(c_1 < 0)
            y.append(y_i)
        return y

class LinearThresh:
    """Class hyperplane classifier."""
    def __init__(self, d):
        self.coefficient = d

    def predict(self, X):
        """Predict labels on data set X."""
        beta = self.coefficient
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            c_1 = np.dot(beta, x_i)
            
            #sm = softmax(np.array([0, c_1]))
            #p = np.random.uniform()
            #y_i = int(p>sm[0]) 
            
            y_i = int(c_1 < 0)
            y.append(y_i)
        return y

