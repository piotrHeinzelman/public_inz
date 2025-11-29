import matplotlib.pyplot as plt
import numpy as np

plt.title('Dobór współczynnika uczenia')
plt.style.use('_mpl-gallery')


x=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
y=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]




x[1]=0.612375 # accuracy 
y[1]=0.020000 # learning rate 
x[2]=0.585500 # accuracy 
y[2]=0.040000 # learning rate 
x[3]=0.722500 # accuracy 
y[3]=0.060000 # learning rate 
x[4]=0.678625 # accuracy 
y[4]=0.080000 # learning rate 
x[5]=0.486750 # accuracy 
y[5]=0.100000 # learning rate 
x[6]=0.147875 # accuracy 
y[6]=0.120000 # learning rate 
x[7]=0.226375 # accuracy 
y[7]=0.140000 # learning rate 
x[8]=0.281625 # accuracy 
y[8]=0.160000 # learning rate 
x[9]=0.328500 # accuracy 
y[9]=0.180000 # learning rate 
x[10]=0.404000 # accuracy 
y[10]=0.200000 # learning rate 
x[11]=0.355875 # accuracy 
y[11]=0.220000 # learning rate 
x[12]=0.253375 # accuracy 
y[12]=0.240000 # learning rate 
x[13]=0.191750 # accuracy 
y[13]=0.260000 # learning rate 
x[14]=0.195000 # accuracy 
y[14]=0.280000 # learning rate 
x[15]=0.173875 # accuracy 
y[15]=0.300000 # learning rate 
x[16]=0.182875 # accuracy 
y[16]=0.320000 # learning rate 
x[17]=0.189625 # accuracy 
y[17]=0.340000 # learning rate 
x[18]=0.290375 # accuracy 
y[18]=0.360000 # learning rate 
x[19]=0.113375 # accuracy 
y[19]=0.380000 # learning rate 
x[20]=0.195000 # accuracy 
y[20]=0.400000 # learning rate 
x[21]=0.113250 # accuracy 
y[21]=0.420000 # learning rate 
x[22]=0.199375 # accuracy 
y[22]=0.440000 # learning rate 
x[23]=0.113125 # accuracy 
y[23]=0.460000 # learning rate 
x[24]=0.113125 # accuracy 
y[24]=0.480000 # learning rate 
x[25]=0.113125 # accuracy 
y[25]=0.500000 # learning rate 
x[26]=0.113125 # accuracy 
y[26]=0.520000 # learning rate 
x[27]=0.113250 # accuracy 
y[27]=0.540000 # learning rate 
x[28]=0.113125 # accuracy 
y[28]=0.560000 # learning rate 
x[29]=0.071875 # accuracy 
y[29]=0.580000 # learning rate 
x[30]=0.046250 # accuracy 
y[30]=0.600000 # learning rate 
x[31]=0.071750 # accuracy 
y[31]=0.620000 # learning rate 
x[32]=0.143375 # accuracy 
y[32]=0.640000 # learning rate 
x[33]=0.098500 # accuracy 
y[33]=0.660000 # learning rate 
x[34]=0.068000 # accuracy 
y[34]=0.680000 # learning rate 
x[35]=0.117500 # accuracy 
y[35]=0.700000 # learning rate 
x[36]=0.111375 # accuracy 
y[36]=0.720000 # learning rate 
x[37]=0.132500 # accuracy 
y[37]=0.740000 # learning rate 
x[38]=0.105500 # accuracy 
y[38]=0.760000 # learning rate 
x[39]=0.038750 # accuracy 
y[39]=0.780000 # learning rate 
x[40]=0.117625 # accuracy 
y[40]=0.800000 # learning rate 
x[41]=0.073625 # accuracy 
y[41]=0.820000 # learning rate 
x[42]=0.129625 # accuracy 
y[42]=0.840000 # learning rate 
x[43]=0.076625 # accuracy 
y[43]=0.860000 # learning rate 
x[44]=0.108000 # accuracy 
y[44]=0.880000 # learning rate 
x[45]=0.110750 # accuracy 
y[45]=0.900000 # learning rate 
x[46]=0.093625 # accuracy 
y[46]=0.920000 # learning rate 
x[47]=0.098500 # accuracy 
y[47]=0.940000 # learning rate 
x[48]=0.096875 # accuracy 
y[48]=0.960000 # learning rate 
x[49]=0.145125 # accuracy 
y[49]=0.980000 # learning rate 
x[50]=0.116625 # accuracy 
y[50]=1.000000 # learning rate 
x[51]=0.106375 # accuracy 
y[51]=1.020000 # learning rate 
x[52]=0.057375 # accuracy 
y[52]=1.040000 # learning rate 
x[53]=0.115125 # accuracy 
y[53]=1.060000 # learning rate 
x[54]=0.131000 # accuracy 
y[54]=1.080000 # learning rate 
x[55]=0.109375 # accuracy 
y[55]=1.100000 # learning rate 
x[56]=0.109250 # accuracy 
y[56]=1.120000 # learning rate 
x[57]=0.110250 # accuracy 
y[57]=1.140000 # learning rate 
x[58]=0.110250 # accuracy 
y[58]=1.160000 # learning rate 
x[59]=0.085875 # accuracy 
y[59]=1.180000 # learning rate 
x[60]=0.108875 # accuracy 
y[60]=1.200000 # learning rate 
x[61]=0.102250 # accuracy 
y[61]=1.220000 # learning rate 
x[62]=0.121750 # accuracy 
y[62]=1.240000 # learning rate 
x[63]=0.100875 # accuracy 
y[63]=1.260000 # learning rate 
x[64]=0.072000 # accuracy 
y[64]=1.280000 # learning rate 
x[65]=0.096250 # accuracy 
y[65]=1.300000 # learning rate 
x[66]=0.085875 # accuracy 
y[66]=1.320000 # learning rate 
x[67]=0.070875 # accuracy 
y[67]=1.340000 # learning rate 
x[68]=0.119875 # accuracy 
y[68]=1.360000 # learning rate 
x[69]=0.111125 # accuracy 
y[69]=1.380000 # learning rate 
x[70]=0.107375 # accuracy 
y[70]=1.400000 # learning rate 
x[71]=0.102500 # accuracy 
y[71]=1.420000 # learning rate 
x[72]=0.131500 # accuracy 
y[72]=1.440000 # learning rate 
x[73]=0.102750 # accuracy 
y[73]=1.460000 # learning rate 
x[74]=0.121500 # accuracy 
y[74]=1.480000 # learning rate 
x[75]=0.101375 # accuracy 
y[75]=1.500000 # learning rate 

















plt.plot(  y, x, '.', color='tab:green')

#plt.legend(['dokładność MLP Java','dokładność CNN Ja']);
plt.xlabel('współczynnik uczenia')
plt.ylabel('dokładność [%]')
plt.savefig( 'dobor_wsp_uczenia.pdf',dpi=400 )
