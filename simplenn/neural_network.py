import numpy as np
import random

class NeuralNetwork ():
    
    def __init__(self, n_hidden = [1]):
        self.n_input = None # It will be defined at first training
        self.n_hidden = n_hidden
        self.n_output = None # It will be defined at first training
        self.weight = list() # It will be defined at first training
        
    def stats(self):
        '''
        
        It prints some infos about the Neural Network.
        
        INPUT:
        
        - None
        
        OUTPUT:
        
        - None
        
        '''
        
        print()
        print("Number of inputs: "+str(self.n_input))
        print("Number of hidden neurons: "+str(self.n_hidden))
        print("Number of outputs: "+str(self.n_output))
        print("The current weights are: "+str(self.weight))
        print()
        
    def train(self, X, y, init_w = "rand"):
        '''
        
        It trains the Neural Network using X as input data and y as
        target data. It doesn't return anything.
        
        INPUT:
        
        - X: numpy array containing the input data;
        - y: numpy array containing the target data;
        - init_w: how should weights be initialised?
          - "rand": randomly
          - an int Q: all Qs
        
        OUTPUT:
        
        - None
        
        '''
        
        # Extract infos from X and y
        n_size, n_feature = X.shape
        n_target = y.shape[1] # The number of outputs
        
        # Define n_input and n_output
        self.n_input = n_feature # There are as many inputs as features
        self.n_output = n_target # There are as many outputs as targets
        
        # Initialize the weights
        if init_w == "rand":
            w = random.random
        else:
            w = lambda: init_w
        
        # First layer of weights
        self.weight.append( [w() 
                            for _ 
                            in range(self.n_input * self.n_hidden[0])
                            ])
        # This for loop only happens if len(n.hidden) > 1
        for i in range(len(self.n_hidden)-1): 
            self.weight.append( [w() 
                                for _ 
                                in range(self.n_hidden[i] * self.n_hidden[i+1])
                                ])
        # Last layer of weights
        self.weight.append( [w() 
                            for _ 
                            in range(self.n_hidden[-1] * self.n_output)
                            ])        
        
        
        
    def fit():
        pass
    

def get_error(output, target):
    '''

    Given two numpy arrays it calculates the squared difference between
    every cell. It returns such difference.
    
    INPUT:
    
    - output: a numpy array
    - target: a numpy array
    
    OUTPUT:
    
    - error: the squared difference of the two arrays
    
    '''
    error = 0
    
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

def import_data(filename, delim = ",", n_target = 1):
    '''
    
    A simple wrapper for Numpy genfromtxt.
    
    INPUT:
    
    - filename: the name of the file containing the data;
    - delim: the delimiter for the data, "," by default - you can change it
             to "\t" for tab-delimited data;
    - n_targets: the number of trailing columns in the dataset that
                 are targets - e.g., predicting weight and gender given age
                 and height would require n_target = 2.
                 
                 
    OUTPUT:
    
    - dataset: numpy array, the whole dataset, X+y;
    - X: numpy array, the dataset without the targets;
    - y: numpy array, the targets
    
    '''
    
    dataset = np.genfromtxt(filename, delimiter = delim)
    nrow, ncol = dataset.shape
    
    X = dataset[0:nrow,0:ncol-n_target]
    y = dataset[0:nrow,ncol-n_target:ncol]
    
    return dataset, X, y