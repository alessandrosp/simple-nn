import simplenn.neural_network as snn

dataset, X, y = snn.import_data("data-AND-operator.txt")
nn = snn.NeuralNetwork(n_hidden = [1,3])

print("Before.")
nn.stats()

print("After.")
nn.train(X,y)
nn.stats()

Z = X.copy()
Z[3,1] = 10
print(X)
print(Z)
print(snn.get_error(X,Z))
