import numpy as np
import copy as c
import random as r

class NeuralNetwork ():
    
    def __init__(self, n_hidden = []):
        self.n_input = None # It will be defined at first training
        self.n_hidden = n_hidden
        self.n_output = None # It will be defined at first training
        self.weight = list() # It will be defined at first training
        self.structure = None
        
    def stats(self):
        '''
        It prints some infos about the Neural Network.
        
        INPUT:
        
        - None
        
        OUTPUT:
        
        - None
        
        '''
        
        print()
        print("Number of inputs: "+str(self.n_input)+" (bias included)")
        print("Number of hidden neurons: "+str(self.n_hidden))
        print("Number of outputs: "+str(self.n_output))
        if len(self.weight) == 0:
            print("The current weights are: None")
        else:
            print("The current weights are: \n")
            for w in self.weight:
                print(w)
                print()
        print()
        
    def train(self, X, y, init_w = "rand", bias = True):
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
        self.n_input = n_feature + 1 # + 1 because bias neuron
        self.n_output = n_target # There are as many outputs as targets
        
        # Now we can use this info to define the NN structure
        self.structure = [self.n_input] + self.n_hidden + [self.n_output]
        
        # Generate the starting weights
        if init_w == "rand":
            for i in range(len(self.structure)-1):
                    self.weight.append(np.random.rand(self.structure[i],
                                                      self.structure[i+1]))  
        else:
            init_w = float(init_w)
            for i in range(len(self.structure)-1):
                    self.weight.append(np.full((self.structure[i],
                                                self.structure[i+1]), 
                                               init_w))
                    
        # Fit the data and check the error
        result = self.fit(X)
        error = get_error(result, y)

        alpha = 0.8
        
        for _ in range(100):
        
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
        '''
        It fits the dataset X on the trained Neural Network. 
        
        INPUT:
        
        - X: numpy array containing the input data;
        - bias: True is a bias neuron (fixed 1) has to be added.
        
        OUTPUT:
        
        - Z: the value of the output layer
        
        '''
        
        # Add bias neuron
        if bias:
            bias_neuron = np.array([[1] for _ in range(X.shape[0])])
            X = np.hstack((X,bias_neuron))        
        
        # Matrix moltiplication
        for layer in range(len(self.structure)-1):
            if layer == 0:
                pre = X
            else:
                pre = Z
            S = pre.dot(np.array(self.weight[layer]))
            Z = S #later func()
            if len(Z.shape) < 2: # If result in a 1d array (it should)
                Z = np.reshape(Z,[Z.shape[0],1])
        return(Z)
            

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

