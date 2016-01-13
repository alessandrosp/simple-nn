import simplenn.neural_network as snn
import simplenn.preprocessing as pp
import numpy as np

dataset, X, y = pp.import_data("data-AND-operator.txt")
nn = snn.NeuralNetwork(n_hidden = [2])

print("Before.")
nn.stats()

print("After.")
nn.train(X, y, init_w = 1)
nn.stats()

print("Data are:")
print(X)

print("\nResults of fit:")
print(np.around(nn.fit(X),0))

print("\nTargets:")
print(y)

print("\nError:")
print(snn.get_error(np.around(nn.fit(X),0),y))


# [DONE] Add bias to every layer!
# Paramter to allow users to choose fun for hidden layer
# Paramter to allow users to choose between 
# regression and categorisation
# http://scikit-learn.org/dev/modules/neural_networks_supervised.html

#if layer == 0:
    #pre = X # Input neurons
#else:
    #pre = Z # Previous layer
    
    #if bias:
        #bias_neuron = np.array([[1] for _ in range(X.shape[0])])
        #X = np.hstack((X,bias_neuron))      