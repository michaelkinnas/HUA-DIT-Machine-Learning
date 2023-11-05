import numpy as np

class LinearRegression:
    def __init__(self):
        w = None        # Weight Matrix
        b = None        # Intercept
         
        # x = None        # Input feature Matrix

    def fit(self, X, y):
        """
        Calculates the w and b parameters for two given matrices.

        Parameters:
        -----------
            X : numpy array
                A numpy array of dimensions N x p containing the dataset features.
            
            y : numpy array
                A numpy array of dimensions N x 1 containing the labels.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError

        if X.T.shape[1] != y.shape[0]:
            raise ValueError        
        
        res = np.dot((1 / (np.dot(X.T, X))), np.dot(X.T, y))

        w = res[:-2]
        b = res[:-1]


