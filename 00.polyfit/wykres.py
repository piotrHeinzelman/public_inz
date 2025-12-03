import matplotlib.pyplot as plt
import numpy as np
plt.title('Regresja liniowa: polyfit 64.000.000')
plt.style.use('_mpl-gallery')


library=[  'C++','Matlab','Java','Pyth []','Pyth np.ary']
library=[None,None,None,None,None]
d0=[None,None,None,None,None]
d1=[None,None,None,None,None]
d2=[None,None,None,None,None]

 

library[0]="Java"
d0[0]=0.451
d1[0]=0.163
d2[0]=0.614

library[1]="Matlab"
d0[1]=0.389
d1[1]=0.273
d2[1]=0.662


library[2]="C++"
d0[2]=0.238
d1[2]=0.382
d2[2]=0.62


library[3]="Python\n[]"
d0[3]=10.73
d1[3]=19.61
d2[3]=30.34

library[4]="Python\nnp.array"
d0[4]=0.29
d1[4]=51.35
d2[4]=51.64

bar_width = 0.2
x = np.arange(len(library))
off=bar_width


 

plt.bar( x-1.0*off, d0,  bar_width, label='czas przygotowania danych', linewidth=.5 )
plt.bar( x-0.0*off, d1,  bar_width, label='wykonanie obliczen', linewidth=.5 )
plt.bar( x+1.0*off, d2,  bar_width, label='całkowity czas wykonania zadania', linewidth=.5 )

plt.legend()
plt.ylabel('time[s]')
plt.xticks(x,library)


plt.savefig( 'wykres_regres.pdf',dpi=400 )
plt.show()
plt.close()



