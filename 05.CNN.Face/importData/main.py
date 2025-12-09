from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

trainX=open( 'trainX', 'wb' )
testX=open( 'testX', 'wb' )
trainY=open( 'trainY', 'wb' )
testY=open( 'testY', 'wb' )
allX=open( 'allX', 'wb' )
allY=open( 'allY', 'wb' )



#my_str = "./PNG/0_1.png"
#img = Image.open( my_str )
#img_array = np.array(img)
#print (img_array.shape)
#newimage=Image.fromarray(img_array,'RGB')
#newimage.show();


for (i) in range(1,21):
   for (k) in range(0,10):
      my_str = "./PNG/"+str(k)+'_'+ str(i) +".png"
      img = Image.open( my_str )
      img_array = np.array(img)
      print(img_array.shape)

      np.array( img_array, np.uint8 ).tofile( allX )
      np.array( k, np.uint8 ).tofile( allY )

      if (i<16):
         np.array( img_array, np.uint8 ).tofile( trainX )
         np.array( k, np.uint8 ).tofile( trainY )
      else:
         np.array( img_array, np.uint8 ).tofile( testX )
         np.array( k, np.uint8 ).tofile( testY )



      # print ( my_str )
      #im = cv2.imread(my_str,mode='RGB')
      #print(type(im))


      # print( people.images[i].shape )
      # 100 , 75
      # print( people.images[i][0][0] )
      #plt.imshow( people.images[i] )
      #plt.show()
      #if ( num>4 ):
      #   np.array( people.images[i]*255, np.uint8 ).tofile( trainX )
      #   np.array( k, np.uint8  ).tofile( trainY )
      #else:
      #   np.array( people.images[i]*255, np.uint8 ).tofile( testX )
      #   np.array( k, np.uint8  ).tofile( testY )


trainX.close()
testX.close()
trainY.close()
testY.close()
allX.close()
allY.close()

   # print( people.images[i].shape )
   # print( people.images[i] )
   # print ( people.target_names[i] )

   #print( person.target_names )
   #print( person )


#for name in people.target_names:
#    print(name)

#lfw_people.data.dtype

#lfw_people.data.shape

#lfw_people.images.shape








