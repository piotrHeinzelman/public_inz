import matplotlib.pyplot as plt
import numpy as np
plt.title('Czas [s] : CNN klasyfikacja twarzy - Alexnet : 227x227x3 500 epoch')
plt.style.use('_mpl-gallery')



#    Matlab Matlab  GPU
x = np.array([ 8,9,10,11,12 ])
y = np.array([ 86, 87, 85, 87, 87 ])
colors = np.array(["blue","blue","blue","blue","white" ])
plt.scatter(x, y, c=colors, label="Matlab - GPU 86 [sek.]")

# Python - to do
x = np.array([13,14,15,16    ])
y = np.array([58,59,59,59  ])
colors = np.array(["red","red","red","red" ]) #"green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
plt.scatter(x, y, c=colors, label="Python - tensorflow bash")


#Java / C++ - to do
x = np.array([5      ])
y = np.array([0  ])
colors = np.array(["green" ]) #"green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
plt.scatter(x, y, c=colors, label="Java/C++")


plt.legend()
plt.savefig( '../000.fig/fig05.pdf',dpi=400 )
plt.show()
plt.close()



