	# from Python - Machine learning i deep learning ISBN: 978-83-283-7001-2
# https://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPClassifier.html

#
#
#  conda install -c conda-forge scikit-learn-intelex
#
#
# Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
# time: 15:33 sek

#
# pip3 install scikit-learn-intelex
#
import time
import datetime
import sys
import struct
import sklearn.neural_network as snn
import numpy as np

from sklearnex import patch_sklearn, config_context
patch_sklearn()


epoch=500
percent=80


def readFileX ( fileName , offset, percent, multi ):
    file=open( fileName, 'rb' )
    file.read( offset )
    data=np.fromfile( fileName, np.uint8, percent*100*784*multi, '', offset )
    data=data.reshape(percent*100*multi, 784)
    data=1-(data/128)
    file.close()
    return data

def readFileY ( fileName , offset, percent, multi ):
    file=open( fileName, 'rb' )
    file.read( offset )
    len=percent*100*multi
    data=np.fromfile( fileName, np.uint8, len, '', offset )

    out=[]
    for i in range ( len ):
        tmp=[0,0,0,0,0,0,0,0,0,0]
        tmp[ data[i]] = 1
        out.append( tmp )
    file.close()
    return out

start=time.time()
trainX = readFileX ('../../data/train-images-idx3-ubyte', 16, percent ,6 )
trainY = readFileY ('../../data/train-labels-idx1-ubyte', 8, percent, 6 )
testX = readFileX ('../../data/t10k-images-idx3-ubyte', 16, percent, 1  )
testY = readFileY ('../../data/t10k-labels-idx1-ubyte', 8, percent, 1 )
end=time.time()
TIME_loadFile=end-start


if (False):
    fig, ax = plt.subplots( nrows=2, ncols=5, sharex=True, sharey=True )
    ax=ax.flatten()
    img = trainX[0].reshape(28,28)
    ax[0].imshow( img, cmap='Greys' )
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()





start=time.time()
net = snn.MLPClassifier( hidden_layer_sizes=(64,64), max_iter=epoch, random_state=1, alpha=0.0,  early_stopping=False, activation='logistic', solver='sgd', learning_rate='constant', learning_rate_init=0.1 )
end=time.time()
TIME_createNet=end-start




start=time.time()

with config_context(target_offload="gpu:0"):
   net.fit( trainX, trainY )

net.partial_fit( trainX, trainY )


end=time.time()
TIME_train=end-start
print("# Python Sklearn Time: " , TIME_train)


start=time.time()
score=net.score( testX, testY )
end=time.time()
TIME_score=end-start



print ("# Python Sklearn loss:", net.loss_ , ", score: " , net.score( testX, testY ), ", predict: ",net.predict(testX[:1]), "epoch:",epoch )
print ("# Load File:",TIME_loadFile  )
print ("# Create Net:",TIME_createNet  )
print ("# Training Net:",TIME_train  )
print ("# Score Time:",TIME_score  )
print ("# Score:",score  )

resultY = net.predict( testX[18] )


