
import tensorflow as tf


import numpy as np
import time
from tensorflow.keras.backend import clear_session
from tensorflow import keras

import numpy as np
from tensorflow.keras.utils import to_categorical

# params
epochs = 500
batch_size = 2000
percent = 80
num_classes = 10
input_shape = (784, 1)



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

timeLoadDataStart=time.time()

trainX = readFileX ('../../data/train-images-idx3-ubyte', 16, percent ,6 )
trainY = readFileY ('../../data/train-labels-idx1-ubyte', 8, percent, 6 )
testX = readFileX ('../../data/t10k-images-idx3-ubyte', 16, percent, 1  )
testY = readFileY ('../../data/t10k-labels-idx1-ubyte', 8, percent, 1 )


trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)

trainX = trainX.astype("float32") # / 255
testX = testX.astype("float32") # / 255
trainX = trainX.reshape(6*percent*100, 784).astype("float32") #/ 255
testX = testX.reshape(1*percent*100, 784).astype("float32") #/ 255

timeLoadDataEnd=time.time()



model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


opt = keras.optimizers.SGD(learning_rate=0.1)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.summary()



timeTrainStart=time.time()

with tf.device('/device:GPU:0'):
    model.fit( trainX, trainY, batch_size=batch_size, epochs=epochs, validation_split=0.0, verbose=0)

timeTrainEnd=time.time()

timeForwardStart=time.time()

with tf.device('/device:GPU:0'):
    result = model.evaluate( testX, testY )
timeForwardEnd=time.time()



loss, acc = model.evaluate( testX, testY )
print("Loss {}, Accuracy {}".format(loss, acc))


print('# Python, MLP: 2x 64 Neu, data size=',percent*600,'' );
print('# accuracy=',result[1])
print('# train: epochs=',epochs);
print('# LoadDataTime=', timeLoadDataEnd-timeLoadDataStart)
print('# trainTime=',timeTrainEnd-timeTrainStart);
print('# propagation time:=',(timeForwardEnd-timeForwardStart) );

