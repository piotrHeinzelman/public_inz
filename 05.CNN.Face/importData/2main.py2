from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.8)
# people = fetch_lfw_people( resize=1.0 )

#for ( i ) in range ( people.target.size ):
#   print( str(i) + " : " + people.target_names[i] )


k0=20
k1=20
k2=20
k3=20
k4=20
k5=20
k6=20
k7=20
k8=20
k9=20
num=0

trainX=open( 'trainX', 'wb' )
testX=open( 'testX', 'wb' )
trainY=open( 'trainY', 'wb' )
testY=open( 'testY', 'wb' )


for ( i ) in range ( 3023 ):
   j = people.target[i]
   k=-1
   if ( j==26  ):
      if (k0>0):
         k=0
         k0=k0-1
         num=k0
   elif ( j==27 ):
      if (k1>0):
         k=1
         k1=k1-1
         num=k1
   elif ( j==28 ):
      if (k2>0):
         k=2
         k2=k2-1
         num=k2
   elif ( j==37 ):
      if (k3>0):
         k=3
         k3=k3-1
         num=k3
   elif ( j==38 ):
      if (k4>0):
         k=4
         k4=k4-1
         num=k4
   elif ( j==42 ):
      if (k5>0):
         k=5
         k5=k5-1
         num=k5
   elif ( j==44 ):
      if (k6>0):
         k=6
         k6=k6-1
         num=k6
   elif ( j==4 ):
      if (k7>0):
         k=7
         k7=k7-1
         num=k7
   elif ( j==53 ):
      if (k8>0):
         k=8
         k8=k8-1
         num=k8
   elif ( j==61 ):
      if (k9>0):
         k=9
         k9=k9-1
         num=k9

   if (k!=-1):
      my_str = str(people.target[i])+'_'+ people.target_names[ people.target[i]] +'_'+str(i)
      print ( my_str )
      plt.imsave( './data/bmp/'+my_str+".bmp" , people.images[i] )
      plt.imsave( './data/jpg/'+my_str+".jpg" , people.images[i] )
      plt.imsave( './data/png/'+my_str+".png" , people.images[i] )

      # print( people.images[i].shape )
      # 100 , 75
      # print( people.images[i][0][0] )
      plt.imshow( people.images[i] )
      plt.show()
      if ( num>4 ):
         np.array( people.images[i]*255, np.uint8 ).tofile( trainX )
         np.array( k, np.uint8  ).tofile( trainY )
      else:
         np.array( people.images[i]*255, np.uint8 ).tofile( testX )
         np.array( k, np.uint8  ).tofile( testY )


trainX.close()
testX.close()
trainY.close()
testY.close()

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








