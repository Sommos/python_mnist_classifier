import requests, gzip, os, hashlib, torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

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
# number of input images per batch
batch_size = 128
# loss function to measure the error between predicted and true labels
loss_function = nn.CrossEntropyLoss()
# optimizer used to update the weights based on the computed gradients
optimizer = torch.optim.Adam(model.parameters())
# lists to store the loss and accuracy for each iteration
losses, accuracies = [], []

# train the model for 10 epochs
for i in (t := trange(10)):
    # sample a random batch from the training set
    sample = np.random.randint(0, X_train.shape[0], size=batch_size)
    # create input and output tensors
    X = torch.tensor(X_train[sample].reshape((-1, 28 * 28))).float()
    Y = torch.tensor(Y_train[sample]).long()
    # clear gradients from previous iteration
    optimizer.zero_grad()
    # compute the prediction for this batch by calling on model
    out = model(X)
    cat = torch.argmax(out, dim=1)
    accuracy = (cat == Y).float().mean()
    # compute loss for this batch
    loss = loss_function(out, Y)
    # compute gradients based on loss
    loss.backward()
    # update weights
    optimizer.step()
    # store current loss and accuracy
    loss, accuracy = loss.item(), accuracy.item()
    # append current loss and accuracy iteration to loss and accuracy lists
    losses.append(loss)
    accuracies.append(accuracy)

    t.set_description("Loss: %.2f | Accuracy: %.2f " % (loss, accuracy))

# plot loss and accuracy
plt.plot(losses)
plt.plot(accuracies)
plt.show()