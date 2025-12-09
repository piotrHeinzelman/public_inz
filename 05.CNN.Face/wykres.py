import matplotlib.pyplot as plt
import numpy as np

plt.title('Trening klasyfikatora DEEP CNN wykrywanie twarzy 250 epok')
plt.style.use('_mpl-gallery')



library=[None,None,None]

d0=[None,None,None]
d1=[None,None,None]
d2=[None,None,None]

#################################


# net: tiny-coco; task:dog1; class: dog score: 0.762755;  prepare detectortime: :2.496382;
# load image time: 0.008481; detect object time: 1.285220
library[ 0 ]="Python\n PyTorch"
d0[0]=0.81


# Matlab, MLP: 2x 64 Neu, epoch=5000, data size=60000, accuracy:0.815000%
library[1]="Matlab GPU"
d0[1]=0.2
d1[1]=0.234
d2[1]=11.8
#d4[0]=0.81


library[2]="Python\n Tensorflow"
d0[2]=0.28
d1[2]=2.83
d2[2]=82.13
#d4[1]=0.53


# net: tiny-yolov4-coco; task:dog1; class: ? score: 0.000000;  prepare detectortime: :0.718696;
# load image time: 0.011418; detect object time: 0.780485

library[ 3 ]="Java"
d0[3]=0.253
d1[3]=5.14
d2[3]=1403


################################


# Define library names
#library = ['Yolo2', 'Yolo3', 'Yolo4', 'Yolo5']

# Number of Enthusiasts for different regions
#enthusiasts_north = [2000, 1500, 2500, 2000]
#enthusiasts_south = [1500, 1300, 2000, 1800]
bar_width = 0.18
x = np.arange( len(library) )
off = bar_width

# Grouped Bar Plot
plt.bar( x-1.0*off  , d0, bar_width, label='wczytanie z pliku+transfer do GPU')
plt.bar( x-0.0*off  , d1, bar_width, label='predykcja F')
plt.bar( x+1.0*off  , d2, bar_width, label='trenowanie modelu')
#plt.bar( x+2.0*off  , d4, bar_width, label='accuracy[%]')

# Adding labels and title
#plt.xlabel('Yolo ')
plt.ylabel('Time[s]')
plt.xticks(x, library) # ,size=5
plt.legend(title='Time[s]')
plt.savefig( 'Pomiary_KlasyfikatorCNN.pdf',dpi=400 )
#plt.show()
#plt.close()



