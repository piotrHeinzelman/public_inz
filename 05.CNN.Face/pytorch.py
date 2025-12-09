#CUDA available. Using GPU acceleration.
#(120, 3, 227, 227)
# CNN 48000 img, epoch: 250
# Epoch:  249
# trainForward:  0.81 s


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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

if torch.cuda.is_available():
  print("CUDA available. Using GPU acceleration.")
  device = "cuda"
else:
  print("CUDA is NOT available. Using CPU for training.")
  device = "cpu"




# params
epochs = 250
num_classes = 10
dataSize=50


def readFileX ( fileName ,  multi ):
    file=open( fileName, 'rb' )
    data=np.fromfile( fileName, np.uint8, num_classes*multi*227*227*3, '' )
    data=data.reshape(num_classes*multi, 3, 227, 227)
    file.close()
    return data

def readFileY ( fileName ,  multi ):
    file=open( fileName, 'rb' )
    len=num_classes*multi
    data=np.fromfile( fileName, np.uint8, len, '' )
    file.close()
    return data


trainX = readFileX ('data/trainX', dataSize*3 )
trainY = readFileY ('data/trainY', dataSize*3 )
testX = readFileX ('data/testX', dataSize  )
testY = readFileY ('data/testY', dataSize  )


trainY = trainY.astype("int")
testY = testY.astype("int")

trainX = trainX.astype("float32")
testX = testX.astype("float32")


class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, padding=0) # , padding=1)
       self.norm1 = nn.BatchNorm2d(96)
       self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

       self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=0)
       self.norm2 = nn.BatchNorm2d(256)
       self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

       self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=0)

       self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=0)

       self.conv5 = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3)
       self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)


       self.fc1 = nn.Linear( 7744 , 128 ) #  self.fc1 = nn.Linear( 64, num_classes) in, out
       self.fc2 = nn.Linear( 128 , num_classes )


   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = self.norm1(x)
       x = self.pool1(x)

       x = F.relu(self.conv2(x))
       x =        self.norm2(x)
       x =        self.pool2(x)

       x = F.relu(self.conv3(x))

       x = F.relu(self.conv4(x))

       x = F.relu(self.conv5(x))
       x =        self.pool5(x)

       x = x.reshape(x.shape[0],-1)
       x = self.fc1(x)
       x = self.fc2(x)
       return x



start2=time.time()
modelGPU = CNN(in_channels=3, num_classes=num_classes)
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
acc = Accuracy(task="multiclass",num_classes=num_classes).to(device)

# Iterate over the dataset batches
model.eval()


dataTest = torch.tensor( trainX , device=device)
targetsTest = torch.tensor(trainY, device=device)

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
print("# timeDataTransfer: ", timeDataTransfer)
print("# timeTrain: ",timeTrain)
print("# Epoch: " , epoch)
print("# Score[0]:", scores[0])
print("# timeForward: ", timeForward)
print("Test accuracy: ",test_accuracy)


