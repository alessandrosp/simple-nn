import numpy as np
import copy as c
import random as r
import math

class NeuralNetwork ():
    
    def __init__(self, n_hidden = []):
        self.n_input = None # It will be defined at first training
        self.n_hidden = n_hidden # Number of neurons per hidden layer
        self.n_output = None # It will be defined at first training
        self.weight = list() # It will be defined at first training
        self.structure = None # The number of neurons per layer
        
    def stats(self):
        """It prints some infos about the Neural Network."""
        
        print("\nNumber of inputs: "+str(self.n_input)+" (bias included)")
        print("Number of hidden neurons: "+str(self.n_hidden))
        print("Number of outputs: "+str(self.n_output))
        if len(self.weight) == 0:
            print("The current weights are: None")
        else:
            print("The current weights are: \n")
            for w in self.weight:
                print(w,"\n")
        print()
        
    def train(self, X, y, init_w = "rand", bias = True):
        """
        It trains the Neural Network using X as input data and y as
        target data. It doesn't return anything.
        
        Args:
            X (numpy.ndarray): an array containing the input data.
            y (numpy.ndarray): an array containing the target data.
            init_w (Optional[double]): the initial value of the weight; if
                nothing is provided, then they are generated randomly between
                -1 and 1.
            bias(Optional[bool]): True is a bias neuron (fixed 1) has to be 
                added; False otherwise. True by default. 
        
        Returns:
            None
        
        """
        
        # Extract infos from X and y
        n_size, n_feature = X.shape
        n_target = y.shape[1] # The number of outputs
        
        # Define n_input and n_output
        self.n_input = n_feature
        self.n_output = n_target # There are as many outputs as targets
        
        # Now we can use this info to define the NN structure
        self.structure = [self.n_input] + self.n_hidden + [self.n_output]
        
        # Generate the starting weights
        if init_w == "rand":
            for i in range(len(self.structure)-1):
                    self.weight.append(np.random.rand(self.structure[i]+1 
                                                      if bias 
                                                      else self.structure[i],
                                                      self.structure[i+1]))  
        else:
            init_w = float(init_w)
            for i in range(len(self.structure)-1):
                    self.weight.append(np.full((self.structure[i]+1
                                                if bias
                                                else self.structure[i],
                                                self.structure[i+1]), 
                                               init_w))
        
                    
        # Fit the data and check the error
        result = self.fit(X)
        error = get_error(result, y)

        alpha = 0.8
        
        for _ in range(1000):
        
            old_weight = c.deepcopy(self.weight)
            
            for i_weight in range(len(self.structure)-1):
                for j in range(self.weight[i_weight].shape[0]):
                    for k in range(self.weight[i_weight].shape[1]):
                        if r.random() <= alpha:
                            self.weight[i_weight][j][k] += (r.random()*2-1)/10
                            
            new_result = self.fit(X)
            new_error = get_error(new_result,y)
            
            if new_error <= error:
                result = new_result
                error = new_error
            else:
                self.weight = c.deepcopy(old_weight)
        
    def fit(self, X, bias = True):
        """
        It fits the dataset X on the trained Neural Network. 
        
        Args:
            X (numpy.ndarray): array containing the input data.
            bias(Optional[bool]): True is a bias neuron (fixed 1) has to be 
                added; False otherwise. True by default.
        
        Returns:
            Z (numpy.ndarray): the value of the output layer
        
        """
        # Temp
        Z = X
        
        # Matrix moltiplication
        for layer in range(len(self.structure)-1):
            # Add bias neuron
            if bias:
                bias_neuron = np.array([[1] for _ in range(Z.shape[0])])
                Z = np.hstack((Z,bias_neuron))              
                
            S = Z.dot(np.array(self.weight[layer]))
            
            #if layer != len(self.structure)-2: # If not the last layer
            
            Z = np.vectorize(sigmoid)(S) # Hidden neurons use sigmoid func
            #else:
            # Z = np.vectorize(step)(S) # Output neurons use step func
                
            if len(Z.shape) < 2: # If result in a 1d array (it should)
                Z = np.reshape(Z,[Z.shape[0],1])
        return(Z)
            

def get_error(output, target):
    """
    Given two numpy arrays it calculates the squared difference between
    every cell. It returns such difference.
    
    Args:
        output (numpy.ndarray): an array containing the values of 
            the output neurons of the neural network.
        target (numpy.ndarray): an array containing the target values
            for a given dataset.
    
    Returns:
        error (double): the squared difference of the two arrays
    
    """
    error = 0.0
    
    if output.shape != target.shape:
        print("Arrays' shape do not match!") # Should throw an error instead
        return None
    
    if (output == target).all():
        return error
    
    nrow, ncol = output.shape
    for i in range(ncol):
        for j in range(nrow):
            error += (output[j,i] - target[j,i])**2
    return error

def sigmoid(x):
    """Simple logistic sigmoid function."""
    return 1 / (1 + math.exp(-x))

def step(x):
    """Simple step function. Note: it returns an int, not a bool."""
    if x >= 0:
        return 1
    else:
        return 0