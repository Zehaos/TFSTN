import numpy as np

MNIST = np.load('data/MNIST.npz')
trainData = MNIST["train"].item()
print trainData