
import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

"""
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
   for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
      print(gpu.device_type)

device_name = tf.test.gpu_device_name()
print(device_name)
"""


# params
epochs = 20
percent = 100
num_classes = 10
input_shape = (28, 28, 1)


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



def CNN():
   NUMBER_OF_CLASSES = 10
   return keras.models.Sequential([
      keras.layers.Input(shape=(28,28,1) ),
      keras.layers.Conv2D(name='conv1', filters=20, kernel_size=(5,5), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation='softmax')
])



start1=time.time()
trainX = readFileX ('../../01.MPL/data/train-images-idx3-ubyte', 16, percent ,6 )
trainY = readFileY ('../../01.MPL/data/train-labels-idx1-ubyte', 8, percent, 6 )
testX = readFileX ( '../../01.MPL/data/t10k-images-idx3-ubyte', 16, percent, 1  )
testY = readFileY ( '../../01.MPL/data/t10k-labels-idx1-ubyte', 8, percent, 1 )


trainX = trainX.astype("float32") # / 255
testX = testX.astype("float32") # / 255

trainY = trainY.astype("int") # / 255
testY = testY.astype("int") # / 255


trainX = trainX.reshape(6*percent*100, 28,28,1).astype("float32") / 255
testX = testX.reshape(1*percent*100, 28,28,1).astype("float32") / 255
end1=time.time()
timeLoadData=end1-start1


model = CNN()
model.summary()

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])


start=time.time()
#with tf.device('/device:GPU:0'):
model.fit(trainX, trainY, epochs=epochs, verbose=0)

end=time.time()
timeTrain=end-start


start3=time.time()
#with tf.device('/device:GPU:0'):
score = model.evaluate(testX, testY, verbose=1 )
end3=time.time()
timeForward=end3-start3

clear_session()



print("# CNN 48000 img, epoch:",epochs)
print("# timeLoadData: ",timeLoadData)
print("# timeTrain: ",timeTrain)
print("# timeForward: ", timeForward)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


