import requests, gzip, os, hashlib, torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import trange

# function to fetch data from the internet
# returns numpy arrays
def fetch(url):
    # compute file path based on url
    fp = os.path.join("test", hashlib.md5(url.encode('utf-8')).hexdigest())
    # if file exists, read it from disk
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    # otherwise download the file from the internet
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)

    # decompress the data and return is as a numpy array
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

# fetch MNIST data
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # first linear layer with 784 input neurons, 128 output neurons and no bias
        self.l1 = nn.Linear(784, 128, bias=False)
        self.act = nn.ReLU()
        # second linear layer with 128 input neurons, 10 output neurons and no bias
        self.l2 = nn.Linear(128, 10, bias=False)
    # method for forward pass of input tensor 'x' through the network
    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

model = NeuralNet()

model(torch.tensor(X_train[0:10].reshape((-1, 28 * 28))).float())