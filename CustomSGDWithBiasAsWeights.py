import numpy as np
import gc

class CustomSGDLinearRegressor:
    """
    A custom class implementation of Stochastic Gradient Descent for Linear Regression. 
    
    Attributes:
    
        w (array(n_features)): The weights associated with each feature to be learnt using train data points.
        
        X (array(n_samples, n_features)): The training data points used for finding optimal weights and bias.
        
        y (array(n_samples)): The real values associated with the train data points which are to be predicted.
        
        learning_rate - learning_rate (float): The amount of gradient to be used to converge to the solution.
        
        k - batch_size (int): The # random batch points to be taken for calculating gradient. 
        
        epochs - epochs (int): The # times whole data to be passed through the model.
    """
    
    def __init__(self, learning_rate=1e-3, epochs=512, batch_size=64):
        """
        The constructor for the class CustomSGDLinearRegressor.
        
        Parameters:
        
            learning_rate (float): The rate of how fast to converge to the solution. 
        
            epochs (int): The number of times whole data to be passed for convergence. 
        
            batch_size (int): The number of random points to be taken for calculating gradient.
        """
        
        # Initializing the attributes of the class.
        self.w = None
        self.X = None
        self.y = None
        self.iters = epochs
        self.k = batch_size
        self.learning_rate = learning_rate
        
    def fit(self, DTrain, y):
        """
        The method used to fit to the data provided.
        
        Parameters:
        
            DTrain (array(n_samples, n_features)): The data points for fitting the model. 
        
            y (array(n_samples)): The real values to be predicted and for fitting the model.
        """
        
        # Preparing the Train data.
        self.X = np.array(DTrain)
        self.X = np.hstack(( self.X, np.ones(( self.X.shape[0], 1 )) ))
        
        self.y = np.array(y).reshape(-1)
        
        self.w = np.random.normal(size = self.X.shape[1])
        
        # Iterating through the complete data, basically number of epochs.
        for _ in range( self.iters ):
            self.findBestWeights()
        
        self.DeAllocateMemory()
        
    def DeAllocateMemory(self):
        self.X = None
        self.y = None
        gc.collect()
    
    def findBestWeights(self):
        """
        The method used for updating the weights based on learning rate.
        
        Parameters:
            None
        """
        
        opt_weights = self.w
        
        r = self.learning_rate
        
        # Selecting the K random batch points and getting the gradient value using the old weights.
        for _ in range( self.iters // self.k ):
            opt_weights = opt_weights + ( r * 2 * self.selectKPointsforWeights( opt_weights ) ) / self.k
            
            # Reducing the learning rate to avoid oscillation and jump over optimal solution.
            r /= 2
        
        self.w = opt_weights
            
    def selectKPointsforWeights(self, weights):
        """
        The method to get the gradient of weights for k random points.
        
        Parameters:
            weights (array(n_features)): The weights associated with each feature.
        
        Returns:
            float: The value of gradient at old weight.
        """
        
        # Generating K random batch indices.
        idxs = np.random.randint(0, self.X.shape[0], size = self.k)
        
        return np.matmul(self.y[idxs] - np.matmul(self.X[idxs], weights), self.X[idxs])
        
    def predict(self, DTest):
        """
        The method used to predict the real value given an unseen data point.
        
        Parameters:
            DTest (array(n_samples, n_features)): The data points for the model has to predict real value.
        
        Returns:
            float: The predicted real value using the optimal weights found by SGD.
        """
        
        # Preparing the Test data.        
        DTest = np.array(DTest)
        DTest = np.hstack(( DTest, np.ones(( DTest.shape[0], 1 )) ))
        
        # Predicting the real value for the test data points.
        return np.round((self.w * DTest).sum(axis=1), 1)
