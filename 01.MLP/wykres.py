import matplotlib.pyplot as plt
import numpy as np

plt.title('Trening klasyfikatora MLP 500 epok')
plt.style.use('_mpl-gallery')


# https://www.mathworks.com/matlabcentral/answers/1755450-how-to-add-numerical-value-in-the-stacked-bar-chart
# https://www.datacamp.com/tutorial/python-bar-plot

library=[None,None,None,None,None,None,None]

d0=[None,None,None,None,None,None,None]
d1=[None,None,None,None,None,None,None]
d2=[None,None,None,None,None,None,None]
d3=[None,None,None,None,None,None,None]

#################################


# net: tiny-coco; task:dog1; class: dog score: 0.762755;  prepare detectortime: :2.496382;
# load image time: 0.008481; detect object time: 1.285220
library[ 0 ]="Python\n PyTorch"
d0[0]=0.31
d1[0]=0.06
d2[0]=1.001 /100
d3[0]=.907
#d4[3]=0.77






# Matlab, MLP: 2x 64 Neu, epoch=5000, data size=60000, accuracy:0.815000%
library[1]="Matlab GPU"
d0[1]=0.639
d1[1]=0.21
d2[1]=20.77 /100
d3[1]=.72
#d4[0]=0.81




library[ 2 ]="Cuda"
d0[2]=0.65
d1[2]=0.04
d2[2]=23.142 / 100
#d4[4]=d2[0]*.0

library[3]="Python\n Tensorflow"
d0[3]=0.172
d1[3]=0.43
d2[3]=33.20 /100
#d4[1]=0.53


# net: tiny-yolov4-coco; task:dog1; class: ? score: 0.000000;  prepare detectortime: :0.718696;
# load image time: 0.011418; detect object time: 0.780485
library[ 4 ]="Python\n Scikit"
d0[4]=0.222
d1[4]=0.091
d2[4]=263.45 /100
d3[4]=.95
#d4[2]=0.94




library[ 5 ]="Java"
d0[5]=0.25
d1[5]=0.45
d2[5]=1728 / 100




library[ 6 ]="C++ Eigen"
d0[6]=4.7
d1[6]=8.13
d2[6]=3373 / 100
#d4[4]=d2[0]*.0




 



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
plt.bar( x+1.0*off  , d2, bar_width, label='trenowanie modelu / 100 F+B')
#plt.bar( x+2.0*off  , d3, bar_width, label='accuracy[%]')

# Adding labels and title
#plt.xlabel('Yolo ')
plt.ylabel('Time[s]')
plt.xticks(x, library) # ,size=5
plt.legend(title='Time[s]')
plt.savefig( 'Pomiary_KlasyfikatorMLP.pdf',dpi=400 )
#plt.show()
#plt.close()



