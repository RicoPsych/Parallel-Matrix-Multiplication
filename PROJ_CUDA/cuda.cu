extern "C" { 
#include "utility.h"
#include "matrix.h"
#include "matrix.c"
}
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SEED 43526454

// 128KB cache

struct Matrix;

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}


__host__
struct Matrix* GpuAllocateMatrix(struct Matrix* matrix){
  struct Matrix* dmatrix = (struct Matrix*)malloc(sizeof(struct Matrix));
  dmatrix->height = matrix->height;
  dmatrix->width = matrix->width;

  float* mat;
  size_t size = sizeof(float) * dmatrix->width * dmatrix->height;
  printf("Allocating %luB on GPU\n",size);
  if (cudaSuccess!=cudaMalloc((void **) &mat, size))
    errorexit("Error allocating memory on the GPU");

  dmatrix->mat = mat;
  return dmatrix;
}

__host__
void GpuDeallocateMatrix(struct Matrix* dmatrix){
  if (cudaSuccess!=cudaFree((dmatrix->mat))) {
    errorexit("Error when deallocating space on the GPU");
  }

  free(dmatrix);
  // return dmatrix;
}
__host__
void GpuCopyMatrix(struct Matrix* to, struct Matrix* from, cudaMemcpyKind kind){
  if (to->width * to->height != from->width * from->height){
    errorexit("Error copying results, size of destination not equal source");
  }
  size_t size = sizeof(float) * from->width * from->height;
  printf("Copying %luB\n",size);

  if (cudaSuccess!=cudaMemcpy(to->mat, from->mat, size, kind)){
    if(kind == cudaMemcpyHostToDevice){
      errorexit("Error copying results Host2Gpu");
    }
    else {
      errorexit("Error copying results Gpu2Host");
    }    
  }
}


// __host__
// void GpuCopyMatrix(struct Matrix to, struct Matrix from, cudaMemcpyKind kind){
//   if (to.width * to.height != from.width * from.height){
//     errorexit("Error copying results, size of destination not equal source");
//   }
//   size_t size = sizeof(float) * from.width * from.height;
//   printf("Copying %lu\n",size);
//   if (cudaSuccess!=cudaMemcpy(to.mat, from.mat, size, kind)){
//     if(kind == cudaMemcpyHostToDevice){
//       errorexit("Error copying results Host2Gpu");
//     }
//     else {
//       errorexit("Error copying results Gpu2Host");
//     }    
//   }
// }


__global__
void KernelMultiplyMatrixBasic(Matrix dm1, Matrix dm2, Matrix dm3){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int matSize = dm3.height * dm3.width;
  if (tid < matSize){
    int row = tid / dm3.width;
    int col = tid % dm3.width;
    dm3.mat[tid] = 0;
    for (int i = 0; i < dm1.width;i++){
        dm3.mat[tid] += dm1.mat[row*dm1.width + i] * dm2.mat[i*dm2.width + col];
    }  
  }
}

__host__
struct Matrix* GpuMultiplyMatrixBasic( Matrix* m1, Matrix* m2 ){
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct Matrix* dm1 = GpuAllocateMatrix(m1);
  struct Matrix* dm2 = GpuAllocateMatrix(m2);
  struct Matrix* dm3 = GpuAllocateMatrix(m3);

  GpuCopyMatrix(dm1,m1,cudaMemcpyHostToDevice);
  GpuCopyMatrix(dm2,m2,cudaMemcpyHostToDevice);

  // printf("h%d:w%d",m3->height,m3->width);

  int threadsinblock = 1024;
  int blocksingrid = m3->height*m3->width / threadsinblock + 1;

  printf("Starting Kernel b=%d,t=%d\n",blocksingrid,threadsinblock);

  
  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  KernelMultiplyMatrixBasic<<<blocksingrid,threadsinblock>>>(*dm1,*dm2,*dm3);
  cudaDeviceSynchronize();
  
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "Unoptimized");

  // printf("Finished Kernel\n");

  GpuCopyMatrix(m3, dm3, cudaMemcpyDeviceToHost);

  // printf("Retrieved results\n");

  GpuDeallocateMatrix(dm1);
  GpuDeallocateMatrix(dm2);
  GpuDeallocateMatrix(dm3);
  return m3;
}


//Optimization with cache lines, transpose 2 matrix so that the kernel oparates on sequential lines in memory.
__global__
void KernelMultiplyMatrix2(Matrix dm1, Matrix dm2, Matrix dm3){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int matSize = dm3.height * dm3.width;
  if (tid < matSize){
    int row = tid / dm3.width;
    int col = tid % dm3.width;
    dm3.mat[tid] = 0;
    for (int i = 0; i < dm1.width;i++){
        dm3.mat[tid] += dm1.mat[row*dm1.width + i] * dm2.mat[row*dm1.width + i];
    }  
  }
}
__host__
struct Matrix* GpuMultiplyMatrix2( Matrix* m1, Matrix* m2 ){
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct Matrix* dm1 = GpuAllocateMatrix(m1);
  struct Matrix* dm2 = GpuAllocateMatrix(tm2);
  struct Matrix* dm3 = GpuAllocateMatrix(m3);

  GpuCopyMatrix(dm1,m1,cudaMemcpyHostToDevice);
  GpuCopyMatrix(dm2,tm2,cudaMemcpyHostToDevice);

  // printf("h%d:w%d",m3->height,m3->width);

  int threadsinblock = 1024;
  int blocksingrid = m3->height*m3->width / threadsinblock + 1;

  printf("Starting Kernel b=%d,t=%d\n",blocksingrid,threadsinblock);

  
  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  KernelMultiplyMatrix2<<<blocksingrid,threadsinblock>>>(*dm1,*dm2,*dm3);
  cudaDeviceSynchronize();
  
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "Opt1");

  // printf("Finished Kernel\n");

  GpuCopyMatrix(m3, dm3, cudaMemcpyDeviceToHost);

  // printf("Retrieved results\n");
  FreeMatrix(tm2);
  GpuDeallocateMatrix(dm1);
  GpuDeallocateMatrix(dm2);
  GpuDeallocateMatrix(dm3);
  return m3;
}

//Tiling optimization
__global__
void KernelMultiplyTiles(Matrix dm1, Matrix dm2, Matrix dm3){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int matSize = dm3.height * dm3.width;
  if (tid < matSize){
    int row = tid / dm3.width;
    int col = tid % dm3.width;
    dm3.mat[tid] = 0;
    for (int i = 0; i < dm1.width;i++){
        dm3.mat[tid] += dm1.mat[row*dm1.width + i] * dm2.mat[row*dm1.width + i];
    }  
  }
}
__host__
struct Matrix* GpuMultiplyMatrixTiles( Matrix* m1, Matrix* m2 ){
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct Matrix* dm1 = GpuAllocateMatrix(m1);
  struct Matrix* dm2 = GpuAllocateMatrix(tm2);
  struct Matrix* dm3 = GpuAllocateMatrix(m3);

  GpuCopyMatrix(dm1,m1,cudaMemcpyHostToDevice);
  GpuCopyMatrix(dm2,tm2,cudaMemcpyHostToDevice);

  // printf("h%d:w%d",m3->height,m3->width);

  int threadsinblock = 1024;
  int blocksingrid = m3->height*m3->width / threadsinblock + 1;

  printf("Starting Kernel b=%d,t=%d\n",blocksingrid,threadsinblock);

  
  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  KernelMultiplyMatrix2<<<blocksingrid,threadsinblock>>>(*dm1,*dm2,*dm3);
  cudaDeviceSynchronize();
  
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "Opt1");

  // printf("Finished Kernel\n");

  GpuCopyMatrix(m3, dm3, cudaMemcpyDeviceToHost);

  // printf("Retrieved results\n");
  FreeMatrix(tm2);
  GpuDeallocateMatrix(dm1);
  GpuDeallocateMatrix(dm2);
  GpuDeallocateMatrix(dm3);
  return m3;
}



int main(int argc,char **argv) {
  srand(SEED);

  int w,h;
  switch (argc)
  {
    case 1:
      w = h = 10;
      break;
    case 2:
      w = h = atoi(argv[1]);
      break;
    case 3:
      w = atoi(argv[1]);
      h = atoi(argv[2]);
      break;
    default:
      printf("Too much arguments\n");
      break;
  }  
  printf("w = %i, h = %i\n",w,h);

  struct Matrix* m1 = CreateMatrix(h,w);
  struct Matrix* m2 = CreateMatrix(w,h); 
  FillMatrix(m1);
  FillMatrix(m2);


  // struct Matrix* dm1 = GpuAllocateMatrix(m1);
  // struct Matrix* dm2 = GpuAllocateMatrix(m2);
  // GpuCopyMatrix(dm1,m1,cudaMemcpyHostToDevice);
  // GpuCopyMatrix(dm2,m2,cudaMemcpyHostToDevice);

  // cudaDeviceSynchronize();
  // PrintMatrix(m1);
  // PrintMatrix(m2);



  struct Matrix* m3 = GpuMultiplyMatrixBasic(m1, m2);
  struct Matrix* m4 = GpuMultiplyMatrix2(m1,m2);
  // PrintMatrix(matrix3);

  // FreeMatrix(matrix3);


  // struct timeval ins__tstart, ins__tstop;
  // gettimeofday(&ins__tstart, NULL);
  

  // gettimeofday(&ins__tstop, NULL);
  // ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);



  FreeMatrix(m1);
  FreeMatrix(m2);
  FreeMatrix(m3);
  FreeMatrix(m4);

  // FreeMatrix(tmatrix2);
}
