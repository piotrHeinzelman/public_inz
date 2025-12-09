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
library[ 0 ]="Python\nPyTorch\nTensor Core"
d0[0]=8.1


# Matlab, MLP: 2x 64 Neu, epoch=5000, data size=60000, accuracy:0.815000%
library[1]="Matlab GPU CUDA"
d0[1]=62.65


library[2]="Python\n Tensorflow GPU"
d0[2]=1816

bar_width = 0.2
x = np.arange( len(library) )
off = bar_width

# Grouped Bar Plot
plt.bar( x-1.0*off  , d0, bar_width, label='czas treningu 250 epok')

# Adding labels and title
#plt.xlabel('Yolo ')
plt.ylabel('Time[s]')
plt.xticks(x, library) # ,size=5
plt.legend(title='Time[s]')
plt.savefig( 'deepCNNFaces.pdf',dpi=400 )
#plt.show()
#plt.close()



