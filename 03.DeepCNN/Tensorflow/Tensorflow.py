import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical





# params
epochs = 50 # 50
percent = 30 # 30
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



def CNN():
   return keras.models.Sequential([
      keras.layers.Input(shape=(240,240,3) ),

      keras.layers.Conv2D(name='conv1', filters=32, kernel_size=(7,7), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

      keras.layers.Conv2D(name='conv2', filters=64, kernel_size=(5,5), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

      keras.layers.Conv2D(name='conv3', filters=128, kernel_size=(3,3), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

      keras.layers.Conv2D(name='conv4', filters=256, kernel_size=(3,3), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),

      keras.layers.Conv2D(name='conv5', filters=256, kernel_size=(1,1), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(1,1)),

      keras.layers.Conv2D(name='conv6', filters=18, kernel_size=(1,1), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2)),

      keras.layers.Conv2D(name='conv7', filters=8, kernel_size=(1,1), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2)),

      keras.layers.Conv2D(name='conv8', filters=6, kernel_size=(1,1), activation='relu' ),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2)),

      keras.layers.Conv2D(name='conv9', filters=2, kernel_size=(1,1), activation='relu' ),
      keras.layers.BatchNormalization(),


      keras.layers.Flatten(),
      keras.layers.Dense(num_classes, activation='softmax')
])



physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
   for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
      print(gpu.device_type)

device_name = tf.test.gpu_device_name()
print(device_name)



start1=time.time()
trainX = readFileX ('../../../../inz_Hidden/SAS/out.bin' )
trainY = readFileY ('../../../../inz_Hidden/SAS/out.class' )


trainX = trainX.astype("float32")
trainY = trainY.astype("int")

trainX = trainX.reshape(percent*8, 240,240,3).astype("float32")

testX = trainX
testY = trainY

end1=time.time()
timeLoadData=end1-start1


model = CNN()
model.summary()

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])


start=time.time()
#with tf.device('/device:CPU:0'):
with tf.device('/device:GPU:0'):
    model.fit(trainX, trainY, epochs=epochs, verbose=0)

end=time.time()
timeTrain=end-start


start3=time.time()
with tf.device('/device:GPU:0'):
    score = model.evaluate(testX, testY, verbose=1 )
end3=time.time()
timeForward=end3-start3




print("# CNN 48000 img, epoch:",epochs)
print("# timeLoadData: ",timeLoadData)
print("# timeTrain: ",timeTrain)
print("# timeForward: ", timeForward)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


clear_session()
