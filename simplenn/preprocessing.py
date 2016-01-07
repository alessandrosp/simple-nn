import numpy as np

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