import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate #Step size for learning
        self.n_iterations = n_iterations #how many times to see the data
        self.weights = None # The importance factors (to be learned)
        self.bias =None    # The personal mood (to be learned)
        
    def fit(self, X,y):
        """Train the perceptron on daya X and labels y.
        X: training samples(like loan applicants)
        Y: correct answeers (0=deny, 1=approve)
        """  
        n_samples,n_features = X.shape
        
        #initialize the weights to small random numbers
        self.weights = np.random.randn(n_features)* 0.01
        self.bias = 0
        
        #Loop over the dataset multiple times
        for _ in range(self.n_iterations):
            #look at each example one by one
            for idx, x_i in enumerate(X):
                #Compute the score (weighted sum +bias)
                linear_output = np.dot(x_i,self.weights) + self.bias
                
                #make a prediction (o or 1)
                y_predicted = self._step(linear_output) #**(check on this _.step)
           
                #if prediction is wrong, adjust weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
            
    def predict(self,X):             #  Predict class labels for samples in X.
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step(linear_output)
    
    def _step(self, x):
        return np.where(x > 0, 1, 0)  #Step activation function: returns 1 if x > 0 else 0.
    
    #later-add error handling   