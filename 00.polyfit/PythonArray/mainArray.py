import numpy as np
import time 
import datetime
import sys

size = 64000000;

x=[0]*size
y=[0]*size
startFillArray = time.time()

for i in range( size ):
    x[i]=0.1*i
    y[i]=0.2*i

endFillArray = time.time()
print ('# fill array time: ' , endFillArray-startFillArray  , '[sek.]' )

w0=0
w1=0
    
xsr=0
ysr=0

start = time.time()

for i in range( size ):
    xsr+=x[i]
    ysr+=y[i]

xsr=xsr/size
ysr=ysr/size

sumTop=0
sumBottom=0

for i in range( size ):
    z=(x[i]-xsr)
    sumTop+=(z*(y[i]-ysr))
    sumBottom+=(z*z)

w1=sumTop/sumBottom
w0=ysr-(w1*xsr)

end = time.time()
d = end-start


print ('# size: ' , size ,' w1: ', w1 , ', w0: ' , w0  )
print ('# time: ' , d  , '[sek.]' )
print ('')  

 

















 
