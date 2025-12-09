
import matplotlib.pyplot as plt
import numpy as np

species = ("Matlab",  "Python \nTensorflow" )
penguin_means = {
    'CPU': (252, 312),
    'GPU': (15.83, 16.4), 
}

x_ = np.arange(len(species))  
x = np.array(['a','b','c'])

width = 0.4  
multiplier = .5

plt, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x_ + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=2)
    ax.color='red';
    multiplier += 1


 
ax.set_xticks(x_ + width, species)
zz = ax.legend(loc='upper left', ncols=2)


#zz1=zz.legend_handles[1]
#print( zz1 )
#zz1.set_color("yellow")
#ax.legend[0].set_facecolor('yellow')

#ax.legend_handles[1].set_facecolor('yellow')
 
 

 
plt.savefig( '../000.fig/zad4.pdf',dpi=400 ) 
plt.show()
plt.close()



