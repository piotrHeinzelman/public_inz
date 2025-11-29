#include <iostream>
#include <ctime>
#include <fstream>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <vector>

using namespace std;


int main() {

    cout << "start" <<"\n";
   long len = 64000000;
   clock_t tableStart = clock();
   double* X=new double[len];
   double* Y=new double[len];
   for ( int i=0; i<len; i++ ){
      X[i]=0.1*i;
      Y[i]=0.2*i;
   }

   clock_t tableEnd = clock();

   clock_t durationTable = tableEnd - tableStart;

   cout << "table loaded: " <<  durationTable/(CLOCKS_PER_SEC/1000)  << "[msek.]\n";

//-- start
  clock_t before = clock();
   double w1=0.0;
   double w0=0.0;
   double xsr=0.0;
   double ysr=0.0;
   for ( int i=0; i<len; i++ ){
      xsr +=  X[i];
      ysr +=  Y[i];
   }

   xsr=xsr / len;
   ysr=ysr / len;

   double sumTop=0.0;
   double sumBottom=0.0;
      for ( int i=0;i<len;i++ ){
       sumTop   += ((X[i]-xsr)*(Y[i]-ysr));
      sumBottom += ((X[i]-xsr)*(X[i]-xsr));
      }
      w1 = sumTop / sumBottom;
      w0 = ysr -(w1 * xsr) ;

  clock_t duration = clock() - before;


  cout << "create table time: " << (float) durationTable / CLOCKS_PER_SEC << " [sek.],  w0: "<<"\n";
  cout << "#  time: " << (float)duration / CLOCKS_PER_SEC << " [sek.]"  << "\n";
  return 0;

}
