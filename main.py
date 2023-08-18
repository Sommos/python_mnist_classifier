import requests, gzip, os, hashlib, torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import trange

# set default tensor type to disable scientific notation
torch.set_printoptions(sci_mode=False)

# function to fetch data from the internet
# returns numpy arrays
def fetch(url):
    # compute file path based on url
    fp = os.path.join("mnist_data", hashlib.md5(url.encode('utf-8')).hexdigest())
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
# X_train: 60000x28x28 array of pixel data for training images
X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
# Y_train: 60000 array of labels for training images
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
# X_test: 10000x28x28 array of pixel data for testing images
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
# Y_test: 10000 array of labels for testing images
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# MODEL
class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # first linear layer with 784 input neurons, 256 output neurons and no bias
        self.l1 = nn.Linear(784, 256, bias=False)
        self.act1 = nn.ReLU()
        # second linear layer with 256 input neurons, 128 output neurons and no bias
        self.l2 = nn.Linear(256, 128, bias=False)
        self.act2 = nn.ReLU()
        # third linear layer with 128 input neurons, 64 output neurons and no bias
        self.l3 = nn.Linear(128, 64, bias=False)
        self.act3 = nn.ReLU()
        # forth linear layer with 64 input neurons, 10 output neurons and no bias
        self.l4 = nn.Linear(64, 10, bias=False)
    # method for forward pass of input tensor 'x' through the network
    def forward(self, x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        x = self.act3(x)
        x = self.l4(x)

        return x

# TRAINING
# check if GPU is available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = NeuralNet().to(device)

# number of input images per batch
batch_size = 128
# loss function to measure the error between predicted and true labels
loss_function = nn.CrossEntropyLoss()
# optimizer used to update the weights based on the computed gradients
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
# old optimizer using adam instead of sgd
# optimizer = torch.optim.Adam(model.parameters())

# lists to store the loss and accuracy for each iteration
losses, accuracies = [], []

# variables for early stopping
best_val_loss = float('inf')
epochs_since_best_val_loss = 0
patience = 10

# train the model for 1000 epochs
for i in (t := trange(1000)):
    # sample a random batch from the training set
    sample = np.random.randint(0, X_train.shape[0], size=batch_size)
    # create input and output tensors
    X = torch.tensor(X_train[sample].reshape((-1, 28 * 28))).float().to(device)
    Y = torch.tensor(Y_train[sample]).long().to(device)
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

    # evaluate the model on the validation set every 10 epochs
    if i % 10 == 0:
        # compute the validation loss
        X_val = torch.tensor(X_test.reshape((-1, 28 * 28))).float().to(device)
        Y_val = torch.tensor(Y_test).long().to(device)
        val_out = model(X_val)
        val_loss = loss_function(val_out, Y_val).item()

        # check if the validation loss is better than the best validation loss seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_best_val_loss = 0
        else:
            epochs_since_best_val_loss += 1

        # stop training if patience epochs have passed since best validation loss
        if epochs_since_best_val_loss >= patience:
            break

# clip the graph to reasonable values for better visualization
plt.ylim(-0.1, 1.1)
# plot loss and accuracy
plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.show()

# EVALUATION
# compute the model accuracy on the test set
Y_test_predictions = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28 * 28))).float().to(device)), dim=1).cpu().numpy()
print((Y_test == Y_test_predictions).mean())