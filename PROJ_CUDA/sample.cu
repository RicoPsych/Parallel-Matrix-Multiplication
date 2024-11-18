#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

//generate elements of sequence – parallel part
__global__ 
void calculate(int *result) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    result[my_index]=my_index;
}


int main(int argc,char **argv) {

    long long result;
    int threadsinblock=1024;
    int blocksingrid=10000;	

    int size = threadsinblock*blocksingrid;
    //memory allocation on host
    int *hresults=(int*)malloc(size*sizeof(int));
    if (!hresults) errorexit("Error allocating memory on the host");	

    int *dresults=NULL;
    //memory allocation on device (GPU)
    if (cudaSuccess!=cudaMalloc((void **)&dresults,size*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //call kernel on GPU – calculation are executed on GPU
    calculate<<<blocksingrid,threadsinblock>>>(dresults);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //copy all elements from device to host
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,size*sizeof(int),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");


    //calculate sum of all elements on CPU side
    result=0;
    for(int i=0;i<size;i++) {
      result = result + hresults[i];
    }

    printf("\nThe final result is %lld\n",result);

    //free resources
    free(hresults);
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");

}
