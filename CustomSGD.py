import pandas as pd
import numpy as np

class CustomSGDLinearRegressor:
    """
    A Custom class implementation of Stochastic Gradient Descent for Linear Regression. 
    
    Attributes:
        learning_rate (float) : The rate of how fast to converge to the solution.
        
        iters (int) : The number of times whole data to be passed for convergence. 
        
        k_rand_points (int) : The number of random points to be taken for calculating gradient. 
    """
    
    def __init__(self, learning_rate, iters, k_rand_points):
        """
        The constructor for the class CustomSGDLinearRegressor.
        
        Parameters:
            learning_rate (float) : The rate of how fast to converge to the solution. 
        
            iters (int) : The number of times whole data to be passed for convergence. 
        
            k_rand_points (int) : The number of random points to be taken for calculating gradient.
        """
        
        # Initializing the attributes of the class.
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.r = learning_rate
        self.k = k_rand_points
        self.iters = iters
        
    def fit(self, DTrain, y):
        """
        The method used to fit to the data provided.
        
        Parameters:
            DTrain (list of lists) : The data points for fitting the model. 
        
            y (list) : The real values to be predicted and for fitting the model.
        
        Returns:
            None
        """
        
        # Preparing the Train data.
        self.X = np.array(DTrain)
        self.y = np.array(y).reshape(-1)
        self.w = np.random.normal(size = self.X.shape[1])
        self.b = np.random.normal()
        
        # Iterating through the data basically number of epochs.
        for _ in range( self.iters // self.k + 1):
            self.findBestBias()
            self.findBestWeights()
    
    def findBestWeights(self):
        """
        The method used for updating the weights based on learning rate.
        
        Parameters:
            None
        
        Returns:
            None
        """
        
        opt_weights = self.w
        r = self.r
        
        # Selecting the K random points and getting the gradient value using the old weights.
        for _ in range(self.iters - 1):
            opt_weights = opt_weights + ( r * 2 * self.selectKPointsforWeights( opt_weights ) ) / self.k
            
            # Reducing the learning rate to avoid oscillation and jumping over optimal soultion.
            r /= 2
        
        self.w = opt_weights
            
    def selectKPointsforWeights(self, weights):
        """
        The method to get the gradient of weights for k random points.
        
        Parameters:
            weights (list) : The weights associated with each feature.
        
        Returns:
            float : The value of gradient at old weight.
        """
        
        # Generating K random indices.
        idxs = np.random.randint(0, self.X.shape[0], size = self.k)
        
        return (self.X[idxs] * (self.y[idxs] - (weights * self.X[idxs]).sum(axis=1) - self.b).reshape(-1,1)).sum(axis=0)
        
    def findBestBias(self):
        """
        The method used for updating the bias based on learning rate.
        
        Parameters:
            None
        
        Returns:
            None
        """
        
        opt_b = self.b
        r = self.r
        
        # Selecting the K random points and getting the gradient value using the old bias.
        for _ in range(self.iters - 1):
            opt_b = opt_b + ( r * 2 * self.selectKPointsforBias( opt_b ) ) / self.k
            
            # Reducing the learning rate to avoid oscillation and jumping over optimal soultion.
            r /= 2
        
        self.b = opt_b
            
    def selectKPointsforBias(self, bias):
        """
        The method to get the gradient of bias for k random points.
        
        Parameters:
            bias (float) : The bias associated with linear combination of weights.
        
        Returns:
            float : The value of gradient at old bias.
        """
        
        # Generating K random indices.
        idxs = np.random.randint(0, self.X.shape[0], size = self.k)
        
        return (self.y[idxs] - (self.w * self.X[idxs]).sum(axis=1) - bias).sum()
        
    def predict(self, DTest):
        """
        The method used to predict the real value given an unseen data point.
        
        Parameters:
            DTest (list of lists) : The data points for the model has to predict real value.
        
        Returns:
            float : The predicted real value using the optimal weights found by SGD.
        """
        
        # Preparing the Test data.
        DTest = np.array(DTest)
               
        # Predicting the real value for the test data points.
        return np.round( (self.w* DTest).sum(axis=1) + self.b, 1 )
