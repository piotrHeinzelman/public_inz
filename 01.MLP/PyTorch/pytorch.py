import numpy as np

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchmetrics import Accuracy


import inspect
import os
import time

if torch.cuda.is_available():
  print("CUDA available. Using GPU acceleration.")
  device = "cuda"
else:
  print("CUDA is NOT available. Using CPU for training.")
  device = "cpu"




# params
epochs = 500
percent = 80
num_classes = 10


def readFileX ( fileName , offset, percent, multi ):
    file=open( fileName, 'rb' )
    file.read( offset )
    data=np.fromfile( fileName, np.uint8, percent*100*784*multi, '', offset )
    data=data.reshape(percent*100*multi, 784)
    data=(data/255)
    file.close()
    return data

def readFileY ( fileName , offset, percent, multi ):
    file=open( fileName, 'rb' )
    file.read( offset )
    len=percent*100*multi
    data=np.fromfile( fileName, np.uint8, len, '', offset )
    file.close()
    return data


start1=time.time()
trainX = readFileX ('../../data/train-images-idx3-ubyte', 16, percent ,6 )
trainY = readFileY ('../../data/train-labels-idx1-ubyte', 8, percent, 6 )
testX = readFileX ('../../data/t10k-images-idx3-ubyte', 16, percent, 1  )
testY = readFileY ('../../data/t10k-labels-idx1-ubyte', 8, percent, 1 )


trainX = trainX.astype("float32")
testX = testX.astype("float32")

trainY = trainY.astype("int")
testY = testY.astype("int")
end1=time.time()
timeLoadData=end1-start1


class MLP(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(MLP, self).__init__()

       self.fc0 = nn.Linear( 784, 64 ) #  self.fc1 = nn.Linear( 64, num_classes) in, out
       self.fc1 = nn.Linear( 64, 64 ) #  self.fc1 = nn.Linear( 64, num_classes) in, out
       self.fc2 = nn.Linear( 64, num_classes )

   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = self.fc0(x)            # Apply fully connected layer
       x = self.fc1(x)            # Apply fully connected layer
       x = self.fc2(x)            # Apply fully connected layer
       return x



start2=time.time()
modelCPU = MLP(in_channels=1, num_classes=10)
model = modelCPU.to(device)



# Define the loss function
criterion = nn.CrossEntropyLoss()


optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.0)
data = torch.tensor( trainX , device=device)
targets = torch.tensor(trainY, device=device)
end2=time.time()
timeDataTransfer=end2-start2




start=time.time()


for epoch in range(epochs):
   scores = model(data)
   loss = criterion(scores, targets)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

end=time.time()
timeTrain=end-start




# Set up of multiclass accuracy metric
acc = Accuracy(task="multiclass",num_classes=10).to(device)

# Iterate over the dataset batches
model.eval()


dataTest = torch.tensor( testX , device=device)
targetsTest = torch.tensor(testY, device=device)

start3=time.time()
with torch.no_grad():
   outputs = model(dataTest)
   _, preds = torch.max(outputs, 1)
   preds = preds.to(device)
   acc(preds, targetsTest)

end3=time.time()
timeForward=end3-start3

test_accuracy = acc.compute()

print("# timeLoadData: ",timeLoadData)
print("# timeDataTransfer: ", timeDataTransfer)
print("# timeTrain: ",timeTrain)
print("# Epoch: " , epoch)
print("# Score[0]:", scores[0])
print("# timeForward: ", timeForward)
print(f"Test accuracy: {test_accuracy}")

print ( model( dataTest[22] ) )







