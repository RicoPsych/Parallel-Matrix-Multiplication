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
#define TILE_SIZE 32 //32

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


//Optimization with cache lines, transpose 2 matrix so that the kernel operates on sequential lines in memory.
__global__
void KernelMultiplyMatrix2(Matrix dm1, Matrix tdm2, Matrix dm3){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int matSize = dm3.height * dm3.width;
  if (tid < matSize){
    int row = tid / dm3.width;
    int col = tid % dm3.width;
    dm3.mat[tid] = 0;
    for (int i = 0; i < dm1.width;i++){
        dm3.mat[tid] += dm1.mat[row*dm1.width + i] * tdm2.mat[col*dm1.width + i];
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
// Shared Memory
//Cache size per SM - 128KB/
//48KB per block?
//


__global__
void KernelMultiplyTiles(Matrix dm1, Matrix tdm2, Matrix dm3){
  __shared__ float dm1tile [TILE_SIZE][TILE_SIZE];
  __shared__ float dm2tile [TILE_SIZE][TILE_SIZE]; //transposedmatrix
  int _row = threadIdx.y + blockIdx.y * blockDim.y;
  int _tile_row = threadIdx.y;
  int _col = threadIdx.x + blockIdx.x * blockDim.x;
  int _tile_col = threadIdx.x;
  // int _rowsize = ( gridDim.x * blockDim.x);

  //tiles num for splitting 1st and 2nd matrix rows(2nd matrix is transposed) 
  int tiles_num = dm1.width/TILE_SIZE; 
  int last_tile_size = dm1.width%TILE_SIZE; 
  
  // use only threads that are in outputmatrix dimensions
  //Threads that are in range of product matrix, there could be more threads than elements in matrix  
  bool inRange = _col < dm3.width && _row < dm3.height; 
  //Threads that are in range of tiles

  float result = 0.0;

  //for each full tile
  int tile_i = 0;
  for (; tile_i < tiles_num; tile_i++){
    //if thread is in 
      __syncthreads();
      int m1id = _row * dm1.width + _tile_col + tile_i * TILE_SIZE;
      int m2id = _col * dm1.width  + _tile_row + tile_i * TILE_SIZE;
      dm1tile[_tile_row][_tile_col] =  dm1.mat[m1id];
      dm2tile[_tile_col][_tile_row] = tdm2.mat[m2id];
      //debug printf
      // if (dm1.mat[m1id] != tdm2.mat[m2id]) printf("tid:%dx%d tile:%d - a[%d][%d]:%f from m1[%d]       b[%d][%d]:%f from m2[%d]\n",   _row , _col , tile_i,  _tile_row , _tile_col ,   dm1.mat[m1id],m1id, _tile_col ,_tile_row , tdm2.mat[m2id],m2id );
      __syncthreads();

    if (inRange){ //Anti deadlock measure
      for (int i = 0; i < TILE_SIZE; i++){
        result += dm1tile[_tile_row][i] * dm2tile[_tile_col][i];
      }  
    }
  }

  __syncthreads();
  //add last tile, copies 0 to shared memory when out of range
  int m1id = _row * dm1.width + _tile_col + tile_i * TILE_SIZE;
  int m2id = _col * dm1.width  + _tile_row + tile_i * TILE_SIZE;
  dm1tile[_tile_row][_tile_col] =  dm1.mat[m1id];
  dm2tile[_tile_col][_tile_row] = tdm2.mat[m2id];
  //debug printf
  // if (dm1.mat[_row * dm1.width + _tile_col + tile_i * TILE_SIZE] != tdm2.mat[_col * dm1.width  + _tile_row + tile_i * TILE_SIZE]) printf("tid:%dx%d tile:%d - a[%d][%d]:%f from m1[%d]       b[%d][%d]:%f from m2[%d]\n",   _row , _col , tile_i, _tile_row , _tile_col , dm1.mat[m1id],m1id, _tile_col ,_tile_row , tdm2.mat[m2id],m2id );
  __syncthreads();
  if (inRange){ //Anti deadlock measure
  for (int i = 0; i < last_tile_size; i++){
    result += dm1tile[_tile_row][i] * dm2tile[_tile_col][i];
  } 
  
    dm3.mat[_row*dm3.width + _col] = result;
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

  // int threadsinblock = 1024; //-> tile size 32x32 TILE_SIZE
  dim3 threadsinblock = dim3(TILE_SIZE, TILE_SIZE, 1);
  dim3 grid = dim3(m3->width/TILE_SIZE+1, m3->height/TILE_SIZE+1 ,1);

  // int blocksingrid = m3->height*m3->width / threadsinblock + 1;

  printf("Starting Kernel g=%dx%d,t=%dx%d\n",m3->width/TILE_SIZE+1,m3->height/TILE_SIZE+1,TILE_SIZE,TILE_SIZE);

  
  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  KernelMultiplyTiles<<<grid,threadsinblock>>>(*dm1,*dm2,*dm3);
  cudaDeviceSynchronize();
  
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "Tiles");

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
  // PrintMatrix(m1);
  // PrintMatrix(m2);

  struct Matrix* m3 = GpuMultiplyMatrixBasic(m1, m2);
  // struct Matrix* m4 = GpuMultiplyMatrix2(m1,m2);
  struct Matrix* m5 = GpuMultiplyMatrixTiles(m1,m2);
  // PrintMatrix(m3);
  // PrintMatrix(m4);
  // PrintMatrix(m5);
  // PrintMatrixPart(m5);
  // PrintMatrixPart(m3);

  // CompareMatrixFindidxAll(m4, m5,0);
  
  // printf("%d\n",CompareMatrix(m4,m3));
  printf("%d\n",CompareMatrix(m5,m3));
  // printf("%d\n",CompareMatrix(m5,m4));


  FreeMatrix(m1);
  FreeMatrix(m2);
  FreeMatrix(m3);
  // FreeMatrix(m4);
  FreeMatrix(m5);

}
