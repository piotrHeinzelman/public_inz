#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string.h>
#include <vector>

using namespace std;


cudaError_t all     ( unsigned int size_bl, unsigned int size_th , unsigned int page, double* results );
cudaError_t sum     ( double* source, double * result, unsigned int size_bl, unsigned int size_th , double multi);
cudaError_t sumMulti( double* ary,    double * avg,    double* ary2, double * avg2, unsigned int sizex, unsigned int sizey, double* value);

__global__ void fillCU(double* ary, double* dest, float multi)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<250000){
       double sum=0.0;
       for (int j=0;j<256;j++){
          double val=(i*256+j)*multi;
          ary[i*256+j]=val;
          sum+=val;
       }
       dest[i]=sum;
    }
}


__global__ void  sumDest(double* source, double* destination, unsigned int vector_size, unsigned int dest_size ) //<<<1,sizex>>>(dev_destination, dev_destination_lev2 );
{
int i = threadIdx.x; // one block
    double sum=0.0;
    if (i<dest_size ){
       for (int j=0;j<vector_size;j++){
          sum+=source[i*vector_size+j];
       }
    }
    destination[i]=sum;
}

/*   vector_size
     -----------
     |  |  |  |   ->   } dest_size
     |  |  |  |   ->
     ----------
*/


__global__ void divVal( double* results, double* sumx, double* sumy ){
    int i = threadIdx.x;
    if (i==0) {
       sumx[0]=sumx[0]/64000000;
       sumy[0]=sumy[0]/64000000;

       results[0]=sumx[0];//xsr
       results[1]=sumy[0];//ysr
    }
}


__global__ void calc( double* results, double* sumT, double* sumB ){
    int i = threadIdx.x;
    if (i==0) {
       results[2]=sumT[0];
       results[3]=sumB[0];
       results[4]=sumT[0]/sumB[0];//W1=sumT/sumB
       results[5]=results[1]-(results[4]*results[0]);//W0=ysr-(W1*xsr)
    }
}


// sumTop    = E (X[i]-xsr )*( Y[i]-ysr )
// sumBottom = E (X[i]-xsr )*( X[i]-xsr )
__global__ void sumMultiCU(double* ary, double* avg, double* ary2, double* avg2, double* dest)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<250000){
       double sum=0.0;
       for (int j=0;j<256;j++){
          double xi   =  ary[i*256+j];
          double xyi  = ary2[i*256+j];
          xi = xi- avg[0];
          xyi=xyi-avg2[0];
          sum+=xi*xyi;
       }
       dest[i]=sum;
    }
}

















int main()
{
    unsigned int const SIZE_TH = 500; //=64*1000 / 10;
    unsigned int const SIZE_BL = 500; //=1*1000 / 10;
    unsigned int const PAGE    = 256;
    double * results = new double[10];

    for (int i=0;i<10;i++){
       results[i]=(double)0.99*i;
    }

    clock_t start = clock();

    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus == cudaSuccess) {
        cudaStatus = all( SIZE_BL, SIZE_TH, PAGE, results);
    }
    printf("\r\nResults[0] (OK): %f \r\n", results[0] );
    printf("\r\nResults[1] (OK): %f \r\n", results[1] );
    printf("\r\nW0: %f \r\n", results[4] );
    printf("\r\nW1: %f \r\n", results[5] );

//    cudaStatus = sumMulti( X, Xsm, X, Xsm, SIZE_BL, SIZE_TH, sumT);
//    cudaStatus = sumMulti( X, Xsm, Y, Xsm, SIZE_BL, SIZE_TH, sumB);


    clock_t end = clock();
    clock_t myTime = end - start;
    printf("\r\ntime: %lu [clocks tick], %ld[msek]\r\n", myTime, (myTime*1000)/CLOCKS_PER_SEC );
    return 0;

}


// --------------- WORKING --------------




















cudaError_t all( unsigned int sizex, unsigned int sizey, unsigned int page, double* results)
{

    double *dev_results = 0;
    double *dev_X = 0;//=new double[sizex*sizey*page];
    double *dev_Y = 0;//=new double[sizex*sizey*page];
    double *dev_tmp = 0;//=new double[sizex*sizey*page];
    double *dev_tmp2 = 0;//=new double[sizex*sizey*page];
    double *dev_sumX = 0;
    double *dev_sumY = 0;
    double *dev_sumT = 0;
    double *dev_sumB = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_results, 10*sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_X, 256*sizex*sizey * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_Y, 256*sizex*sizey * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_tmp, sizex*sizey * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc4 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_tmp2, sizex * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc5 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_sumX, 1 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc6 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_sumY, 1 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc7 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_sumT, 1 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc7 failed!");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_sumB, 1 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc7 failed!");
        goto Error;
    }



    // FILL dev_X
    fillCU<<<sizey, sizex>>>(dev_X, dev_tmp, 0.1);
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "synch0 failed!"); goto Error; }

    sumDest<<<1,sizex>>>(dev_tmp, dev_tmp2, sizey, sizex );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s1 failed!"); goto Error; }

    sumDest<<<1,1>>>(dev_tmp2, dev_sumX, sizex, 1 );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s2 failed!"); goto Error; }

    // FILL dev_Y
    fillCU<<<sizey, sizex>>>(dev_Y, dev_tmp, 0.2);
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s3 failed!"); goto Error; }

    sumDest<<<1,sizex>>>(dev_tmp, dev_tmp2, sizey, sizex );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s4 failed!"); goto Error; }

    sumDest<<<1,1>>>(dev_tmp2, dev_sumY, sizex, 1 );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s5 failed!"); goto Error; }

    // dev_sumX = sum/64000000; dev sumY = sum/64000000; result[0]=srX, result[1]=srY
    divVal<<<1,2>>>(dev_results, dev_sumX, dev_sumY );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s6 failed!"); goto Error; }




    // SUM (X-xsr)*(Y-ysr)
    sumMultiCU<<<sizey, sizex>>>(dev_X, dev_sumX, dev_Y, dev_sumY, dev_tmp);
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s7 failed!"); goto Error; }

    sumDest<<<1,sizex>>>(dev_tmp, dev_tmp2, sizey, sizex );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s8 failed!"); goto Error; }

    sumDest<<<1,1>>>(dev_tmp2, dev_sumT, sizex, 1 );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s9 failed!"); goto Error; }

    // SUM (X-xsr)*(Y-ysr)
    sumMultiCU<<<sizey, sizex>>>(dev_X, dev_sumX, dev_X, dev_sumX, dev_tmp);
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s7 failed!"); goto Error; }

    sumDest<<<1,sizex>>>(dev_tmp, dev_tmp2, sizey, sizex );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s8 failed!"); goto Error; }

    sumDest<<<1,1>>>(dev_tmp2, dev_sumB, sizex, 1 );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s9 failed!"); goto Error; }

    calc<<<1,1>>>(dev_results, dev_sumT, dev_sumB );
    cudaStatus = cudaDeviceSynchronize(); if (cudaStatus != cudaSuccess) { fprintf(stderr, "s10 failed!"); goto Error; }


/*
    sumDest<<<1,sizex>>>(dev_destination, dev_destination_lev2, sizey, sizex );
    cudaStatus = cudaDeviceSynchronize();
    sumDest<<<1,1>>>(dev_destination_lev2, dev_value, sizex, 1 );
    cudaStatus = cudaDeviceSynchronize();
*/

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, 10 * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
     }

Error:
    cudaFree(dev_results);
    cudaFree(dev_X);
    cudaFree(dev_Y);
    cudaFree(dev_tmp);
    cudaFree(dev_tmp2);
    cudaFree(dev_sumX);
    cudaFree(dev_sumY);

End:
    return cudaStatus;

}
