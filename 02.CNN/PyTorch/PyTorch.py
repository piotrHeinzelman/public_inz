# code from: https://www.datacamp.com/tutorial/pytorch-cnn-tutorial

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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"



if torch.cuda.is_available():
  print("CUDA available. Using GPU acceleration.")
  device = "cuda"
else:
  print("CUDA is NOT available. Using CPU for training.")
  device = "cpu"




# params
epochs = 20
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

trainX = trainX.reshape(6*percent*100, 1, 28,28).astype("float32") # / 255
testX = testX.reshape(1*percent*100, 1, 28,28).astype("float32") # / 255

end1=time.time()
timeLoadData=end1-start1




class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=5, padding=1)
       self.norm  = nn.BatchNorm2d(20)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear( 3380 , num_classes ) #  self.fc1 = nn.Linear( 64, num_classes) in, out


   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.norm(x)           # Apply Normalization
       x = self.pool(x)           # Apply max pooling
       x = x.reshape(x.shape[0], -1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       return x




start2=time.time()
modelGPU = CNN(in_channels=1, num_classes=10)
model = modelGPU.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
#   print( loss )

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

print("# CNN 48000 img, epoch:",epochs)
print("# timeLoadData: ",timeLoadData)
print("# timeDataTransfer: ", timeDataTransfer)
print("# timeTrain: ",timeTrain)
print("# Epoch: " , epoch)
print("# Score[0]:", scores[0])
print("# timeForward: ", timeForward)
print("Test accuracy: ",test_accuracy)


