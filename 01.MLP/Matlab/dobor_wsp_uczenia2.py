import matplotlib.pyplot as plt
import numpy as np

plt.title('Dobór współczynnika uczenia')
plt.style.use('_mpl-gallery')


x=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]
y=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None]







x[1]=0.334875 # accuracy 
y[1]=0.004000 # learning rate 
x[2]=0.445875 # accuracy 
y[2]=0.008000 # learning rate 
x[3]=0.500750 # accuracy 
y[3]=0.012000 # learning rate 
x[4]=0.525875 # accuracy 
y[4]=0.016000 # learning rate 
x[5]=0.536250 # accuracy 
y[5]=0.020000 # learning rate 
x[6]=0.631625 # accuracy 
y[6]=0.024000 # learning rate 
x[7]=0.639125 # accuracy 
y[7]=0.028000 # learning rate 
x[8]=0.559750 # accuracy 
y[8]=0.032000 # learning rate 
x[9]=0.659250 # accuracy 
y[9]=0.036000 # learning rate 
x[10]=0.589750 # accuracy 
y[10]=0.040000 # learning rate 
x[11]=0.668125 # accuracy 
y[11]=0.044000 # learning rate 
x[12]=0.672625 # accuracy 
y[12]=0.048000 # learning rate 
x[13]=0.566875 # accuracy 
y[13]=0.052000 # learning rate 
x[14]=0.678750 # accuracy 
y[14]=0.056000 # learning rate 
x[15]=0.643125 # accuracy 
y[15]=0.060000 # learning rate 
x[16]=0.686875 # accuracy 
y[16]=0.064000 # learning rate 
x[17]=0.601375 # accuracy 
y[17]=0.068000 # learning rate 
x[18]=0.714750 # accuracy 
y[18]=0.072000 # learning rate 
x[19]=0.602625 # accuracy 
y[19]=0.076000 # learning rate 
x[20]=0.724125 # accuracy 
y[20]=0.080000 # learning rate 
x[21]=0.668750 # accuracy 
y[21]=0.084000 # learning rate 
x[22]=0.559125 # accuracy 
y[22]=0.088000 # learning rate 
x[23]=0.708000 # accuracy 
y[23]=0.092000 # learning rate 
x[24]=0.447000 # accuracy 
y[24]=0.096000 # learning rate 
x[25]=0.653375 # accuracy 
y[25]=0.100000 # learning rate 
x[26]=0.697625 # accuracy 
y[26]=0.104000 # learning rate 
x[27]=0.563500 # accuracy 
y[27]=0.108000 # learning rate 
x[28]=0.721125 # accuracy 
y[28]=0.112000 # learning rate 
x[29]=0.106750 # accuracy 
y[29]=0.116000 # learning rate 
x[30]=0.125750 # accuracy 
y[30]=0.120000 # learning rate 
x[31]=0.169375 # accuracy 
y[31]=0.124000 # learning rate 
x[32]=0.103250 # accuracy 
y[32]=0.128000 # learning rate 
x[33]=0.555375 # accuracy 
y[33]=0.132000 # learning rate 
x[34]=0.635000 # accuracy 
y[34]=0.136000 # learning rate 
x[35]=0.612375 # accuracy 
y[35]=0.140000 # learning rate 
x[36]=0.434125 # accuracy 
y[36]=0.144000 # learning rate 
x[37]=0.321250 # accuracy 
y[37]=0.148000 # learning rate 
x[38]=0.421625 # accuracy 
y[38]=0.152000 # learning rate 
x[39]=0.546375 # accuracy 
y[39]=0.156000 # learning rate 
x[40]=0.362500 # accuracy 
y[40]=0.160000 # learning rate 
x[41]=0.158125 # accuracy 
y[41]=0.164000 # learning rate 
x[42]=0.451750 # accuracy 
y[42]=0.168000 # learning rate 
x[43]=0.280500 # accuracy 
y[43]=0.172000 # learning rate 
x[44]=0.419875 # accuracy 
y[44]=0.176000 # learning rate 
x[45]=0.380625 # accuracy 
y[45]=0.180000 # learning rate 
x[46]=0.306625 # accuracy 
y[46]=0.184000 # learning rate 
x[47]=0.293500 # accuracy 
y[47]=0.188000 # learning rate 
x[48]=0.381375 # accuracy 
y[48]=0.192000 # learning rate 
x[49]=0.379125 # accuracy 
y[49]=0.196000 # learning rate 
x[50]=0.261875 # accuracy 
y[50]=0.200000 # learning rate 
x[51]=0.361875 # accuracy 
y[51]=0.204000 # learning rate 
x[52]=0.271125 # accuracy 
y[52]=0.208000 # learning rate 
x[53]=0.418250 # accuracy 
y[53]=0.212000 # learning rate 
x[54]=0.183625 # accuracy 
y[54]=0.216000 # learning rate 
x[55]=0.186250 # accuracy 
y[55]=0.220000 # learning rate 
x[56]=0.367000 # accuracy 
y[56]=0.224000 # learning rate 
x[57]=0.204000 # accuracy 
y[57]=0.228000 # learning rate 
x[58]=0.330250 # accuracy 
y[58]=0.232000 # learning rate 
x[59]=0.498500 # accuracy 
y[59]=0.236000 # learning rate 
x[60]=0.231375 # accuracy 
y[60]=0.240000 # learning rate 
x[61]=0.355875 # accuracy 
y[61]=0.244000 # learning rate 
x[62]=0.185500 # accuracy 
y[62]=0.248000 # learning rate 
x[63]=0.318250 # accuracy 
y[63]=0.252000 # learning rate 
x[64]=0.227125 # accuracy 
y[64]=0.256000 # learning rate 
x[65]=0.087250 # accuracy 
y[65]=0.260000 # learning rate 
x[66]=0.328000 # accuracy 
y[66]=0.264000 # learning rate 
x[67]=0.229375 # accuracy 
y[67]=0.268000 # learning rate 
x[68]=0.185750 # accuracy 
y[68]=0.272000 # learning rate 
x[69]=0.202000 # accuracy 
y[69]=0.276000 # learning rate 
x[70]=0.208000 # accuracy 
y[70]=0.280000 # learning rate 
x[71]=0.399500 # accuracy 
y[71]=0.284000 # learning rate 
x[72]=0.311250 # accuracy 
y[72]=0.288000 # learning rate 
x[73]=0.223250 # accuracy 
y[73]=0.292000 # learning rate 
x[74]=0.194625 # accuracy 
y[74]=0.296000 # learning rate 
x[75]=0.183500 # accuracy 
y[75]=0.300000 # learning rate 
x[76]=0.247250 # accuracy 
y[76]=0.304000 # learning rate 
x[77]=0.176500 # accuracy 
y[77]=0.308000 # learning rate 
x[78]=0.270875 # accuracy 
y[78]=0.312000 # learning rate 
x[79]=0.199750 # accuracy 
y[79]=0.316000 # learning rate 
x[80]=0.170250 # accuracy 
y[80]=0.320000 # learning rate 
x[81]=0.113250 # accuracy 
y[81]=0.324000 # learning rate 
x[82]=0.239125 # accuracy 
y[82]=0.328000 # learning rate 
x[83]=0.212875 # accuracy 
y[83]=0.332000 # learning rate 
x[84]=0.287000 # accuracy 
y[84]=0.336000 # learning rate 
x[85]=0.190625 # accuracy 
y[85]=0.340000 # learning rate 
x[86]=0.280125 # accuracy 
y[86]=0.344000 # learning rate 
x[87]=0.204125 # accuracy 
y[87]=0.348000 # learning rate 
x[88]=0.191000 # accuracy 
y[88]=0.352000 # learning rate 
x[89]=0.190125 # accuracy 
y[89]=0.356000 # learning rate 
x[90]=0.173875 # accuracy 
y[90]=0.360000 # learning rate 
x[91]=0.351750 # accuracy 
y[91]=0.364000 # learning rate 
x[92]=0.173500 # accuracy 
y[92]=0.368000 # learning rate 
x[93]=0.202750 # accuracy 
y[93]=0.372000 # learning rate 
x[94]=0.273000 # accuracy 
y[94]=0.376000 # learning rate 
x[95]=0.143875 # accuracy 
y[95]=0.380000 # learning rate 
x[96]=0.113125 # accuracy 
y[96]=0.384000 # learning rate 
x[97]=0.113250 # accuracy 
y[97]=0.388000 # learning rate 
x[98]=0.201250 # accuracy 
y[98]=0.392000 # learning rate 
x[99]=0.113750 # accuracy 
y[99]=0.396000 # learning rate 
x[100]=0.113125 # accuracy 
y[100]=0.400000 # learning rate 







plt.plot(  y, x, '.', color='tab:green')

#plt.legend(['dokładność MLP Java','dokładność CNN Ja']);
plt.xlabel('współczynnik uczenia')
plt.ylabel('dokładność [%]')
plt.savefig( 'dobor_wsp_uczenia2.pdf',dpi=400 )
