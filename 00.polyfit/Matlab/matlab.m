size = 64000000;


prepareStart = datetime('now');
x=1:size;
y=1:size;

for i=1:size
	   x(i)=0.1*i;
	   y(i)=0.2*i;
end
prepareEnd = datetime('now');


w1 = 0.0; w0 = 0.0;

% start
timeStart = datetime('now');
xsr = 0.0;  ysr = 0.0;

for i=1:size
    xsr = xsr+x(i);
    ysr = xsr+y(i);
end
xsr = xsr / size;
ysr = ysr / size;

w1 = 0.0; w0 = 0.0;
sumTop = 0.0; sumBottom = 0.0;

for i=1:size
    sumTop = sumTop + ((x(i) - xsr) * (y(i) - ysr));
    sumBottom = sumBottom + ((x(i) - xsr) * (x(i) - xsr));
end    
w1 = sumTop / sumBottom;
w0 = ysr - w1 * xsr;

%  -- end
timeEnd = datetime('now');
TIME_=seconds(duration( timeEnd-timeStart ));
TIME_TAB=seconds(duration( prepareEnd-prepareStart ));

fprintf('czas przygotowania tablic (Matlab): %f\n' , TIME_TAB );     
fprintf('czas obliczania regresji (Matlab): %f\n' , TIME_ );
