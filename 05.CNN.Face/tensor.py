#time  1816 s

import tensorflow as tf
import numpy as np
from tensorflow import keras

import os
import time

device_name = tf.test.gpu_device_name()
print(device_name)

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
   for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)

# params
epochs = 250
num_classes = 10
dataSize=5


def readFileX ( fileName ,  multi ):
    file=open( fileName, 'rb' )
    data=np.fromfile( fileName, np.uint8, num_classes*multi*227*227*3, '' )
    data=data.reshape(num_classes*multi, 227, 227, 3)
    file.close()
    return data

def readFileY ( fileName ,  multi ):
    file=open( fileName, 'rb' )
    len=num_classes*multi
    data=np.fromfile( fileName, np.uint8, len, '' )
    file.close()
    return data

def AlexNet():
   NUMBER_OF_CLASSES = 10
   return keras.models.Sequential([
      keras.layers.Input(shape=( 227, 227, 3 )),

      keras.layers.Conv2D(name='conv1', filters=96, kernel_size=(11,11), activation='relu', strides=(4,4) ), 
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

      keras.layers.Conv2D(name='conv2', filters=256, kernel_size=(5,5), activation='relu', strides=(1,1), padding='valid' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),


      keras.layers.Conv2D(name='conv3', filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid' ),

      keras.layers.Conv2D(name='conv4', filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid' ),

      keras.layers.Conv2D(name='conv5', filters=16, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid' ),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

      keras.layers.Flatten(),
      keras.layers.Dense(7744, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
])

trainX = readFileX ('data/trainX', dataSize*3 )
trainY = readFileY ('data/trainY', dataSize*3 )
testX = readFileX ('data/testX', dataSize  )
testY = readFileY ('data/testY', dataSize  )


trainY = trainY.astype("int")
testY = testY.astype("int")

model = AlexNet()

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

start=time.time()

with tf.device('/device:GPU:0'):
    model.fit(trainX, trainY, epochs=epochs, verbose=0)

end=time.time()
d=end-start


print("# Python Tensorflow Time: " , d)

score = model.evaluate(trainX, trainY, verbose=0 )
print("Train loss:", score[0])
print("Train accuracy:", score[1])


score = model.evaluate(testX, testY, verbose=0 )
print("Test loss:", score[0])
print("Test accuracy:", score[1])

