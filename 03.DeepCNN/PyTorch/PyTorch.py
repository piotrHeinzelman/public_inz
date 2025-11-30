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


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]="1"

if torch.cuda.is_available():
  print("CUDA available. Using GPU acceleration.")
  device = "cuda"
else:
  print("CUDA is NOT available. Using CPU for training.")
  device = "cpu"




# params
epochs = 50
percent = 30
num_classes = 2


def readFileX ( fileName ):
    file=open( fileName, 'rb' )
    data=np.fromfile( fileName, np.uint8, percent*8*240*240*3, '')
    data=data.reshape(percent*8, 240*240*3)
    data=(data/255)
    file.close()
    return data

def readFileY ( fileName ):
    file=open( fileName, 'rb' )
    len=percent*8
    data=np.fromfile( fileName, np.uint8, len, '' )
    file.close()
    return data


start1=time.time()

trainX = readFileX ('../../../../inz_Hidden/SAS/out.bin' )
trainY = readFileY ('../../../../inz_Hidden/SAS/out.class' )

trainX = trainX.astype("float32")
trainY = trainY.astype("int")

trainX = trainX.reshape(percent*8, 3, 240,240).astype("float32") # / 255
#trainY = trainY.reshape(percent*8)

print(trainX.shape)
print(trainY.shape)
print(trainY[0])


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
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=3) # , padding=1)
       self.norm1 = nn.BatchNorm2d(32)
       self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=3)
       self.norm2 = nn.BatchNorm2d(64)
       self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
       self.norm3 = nn.BatchNorm2d(128)
       self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
       self.norm4 = nn.BatchNorm2d(256)
       self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

       self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
       self.norm5 = nn.BatchNorm2d(256)
       self.pool5 = nn.MaxPool2d(kernel_size=1)


       self.conv6 = nn.Conv2d(in_channels=256, out_channels=18, kernel_size=1)
       self.norm6 = nn.BatchNorm2d(18)
       self.pool6 = nn.MaxPool2d(kernel_size=2)

       self.conv7 = nn.Conv2d(in_channels=18, out_channels=8, kernel_size=1)
       self.norm7 = nn.BatchNorm2d(8)
       self.pool7 = nn.MaxPool2d(kernel_size=2)

       self.conv8 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=1)
       self.norm8 = nn.BatchNorm2d(6)
       self.pool8 = nn.MaxPool2d(kernel_size=2)

       self.conv9 = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=1)
       self.norm9 = nn.BatchNorm2d(2)
       self.pool9 = nn.MaxPool2d(kernel_size=1)



       # 2nd convolutional layer
       # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear( 2 , num_classes ) #  self.fc1 = nn.Linear( 64, num_classes) in, out
#       self.sm1 =  nn.Softmax(1)

#    \item inputLayer(240, 240, 3)
#    \item convolution2dLayer(7,32)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2, Stride=2)

#    \item convolution2dLayer(5,64)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2, Stride=2)

#    \item convolution2dLayer(3,128)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2, Stride=2)

#    \item convolution2dLayer(3,256)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2, Stride=2)

#    \item convolution2dLayer(1,256)
#    \item batchNormalizationLayer

#    \item convolution2dLayer(1,18, Stride=2)
#    \item batchNormalizationLayer
#    \item reluLayer(Name="Lay")
#    \item maxPooling2dLayer(2)

#    \item convolution2dLayer(1,8, Stride=2)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2)

#    \item convolution2dLayer(1,6, Stride=2)
#    \item batchNormalizationLayer
#    \item reluLayer
#    \item maxPooling2dLayer(2)

#    \item convolution2dLayer(1,2)
#    \item batchNormalizationLayer
#    \item reluLayer

#    \item flattenLayer
#    \item softmaxLayer



   def forward(self, x):
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))   # Apply first convolution and ReLU activation
       x = self.norm1(x)           # Apply Normalization
       x = self.pool1(x)           # Apply max pooling

       x = F.relu(self.conv2(x))
       x =        self.norm2(x)
       x =        self.pool2(x)

       x = F.relu(self.conv3(x))
       x =        self.norm3(x)
       x =        self.pool3(x)

       x = F.relu(self.conv4(x))
       x =        self.norm4(x)
       x =        self.pool4(x)

       x = F.relu(self.conv5(x))
       x =        self.norm5(x)
       x =        self.pool5(x)

       x = F.relu(self.conv6(x))
       x =        self.norm6(x)
       x =        self.pool6(x)

       x = F.relu(self.conv7(x))
       x =        self.norm7(x)
       x =        self.pool7(x)

       x = F.relu(self.conv8(x))
       x =        self.norm8(x)
       x =        self.pool8(x)

       x = F.relu(self.conv9(x))
       x =        self.norm9(x)
       x =        self.pool9(x)
       x = x.reshape(x.shape[0],-1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
#       x = self.sm1(x)
       return x




start2=time.time()
modelGPU = CNN(in_channels=3, num_classes=2)
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
acc = Accuracy(task="multiclass",num_classes=2).to(device)

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
print("# timeLoadData: ",timeLoadData)
print("# timeDataTransfer: ", timeDataTransfer)
print("# timeTrain: ",timeTrain)
print("# Epoch: " , epoch)
print("# Score[0]:", scores[0])
print("# timeForward: ", timeForward)
print("Test accuracy: ",test_accuracy)


