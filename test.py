import simplenn.neural_network as snn
import simplenn.preprocessing as pp
import numpy as np

dataset, X, y = pp.import_data("data-AND-operator.txt")
nn = snn.NeuralNetwork(n_hidden = [])

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
