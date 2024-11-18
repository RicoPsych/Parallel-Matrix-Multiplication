#include "utility.h"
#include "matrix.c"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define SEED 43526454

__host__
void errorexit(const char *s) {
    printf("\n%s",s);	
    exit(EXIT_FAILURE);	 	
}

// __global__
// void MatrixMultiply1(float** numbers,float** results){
//   int my_index=blockIdx.x*blockDim.x+threadIdx.x;
//   long inputArgument = numbers[my_index];
//   int is_prime = 1;
//   for(long mine = 2; mine < inputArgument/2 && is_prime == 1; mine+=1){
//     if((inputArgument % mine == 0 && mine!=1 && mine != inputArgument) || inputArgument == 1 || inputArgument == 0){
//       is_prime = 0;
//     }
//   }
//   results[my_index] = is_prime;
//   //if (is_prime)
//     //printf("%lu\n",inputArgument);
//   //return is_prime;
// }

__host__
float* AllocateAndCopyMatrixGPU(struct Matrix* matrix){
  float* gpu_matrix = NULL;
  int height = matrix->height;
  int width = matrix->width;
  if (cudaSuccess!=cudaMalloc( (void **)&gpu_matrix,height*width*sizeof(float)))
    errorexit("Error allocating memory on the GPU");

  for(int i = 0; i < height; i++){
    if (cudaSuccess!=cudaMemcpy(gpu_matrix+(i*width), matrix->mat[i], width*sizeof(float), cudaMemcpyHostToDevice))
      errorexit("Error copying results");
  }

  return gpu_matrix;
}

__host__
struct Matrix* CopyMatrixFromGPU(float* gpu_matrix, struct Matrix* matrix){
  int height = matrix->height;
  int width = matrix->width;
    
  for(int i = 0; i < height; i++){
    if (cudaSuccess!=cudaMemcpy(matrix->mat[i],gpu_matrix+(i*width), width*sizeof(float), cudaMemcpyDeviceToHost))
      errorexit("Error copying results");
  }

  return matrix;
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

  struct Matrix* matrix1 = CreateMatrix(h,w);
  struct Matrix* matrix2 = CreateMatrix(w,h); 
  FillMatrix(matrix1);
  // FillMatrix(matrix2);
  PrintMatrix(matrix1);
  // PrintMatrix(matrix2);
  struct Matrix* tmatrix2 = TransposeMatrix(matrix2); 
  
  int matrices_sizes[4] = {matrix1->height,matrix1->width, tmatrix2->height,tmatrix2->width}; //Height,Widths pairs;

  
  float *gpu_matrix1=AllocateAndCopyMatrixGPU(matrix1);
  float *gpu_matrix2=NULL;
  float *gpu_result=NULL;

  CopyMatrixFromGPU(gpu_matrix1,matrix2);
  PrintMatrix(matrix2);

  // struct timeval ins__tstart, ins__tstop;
  // gettimeofday(&ins__tstart, NULL);
  
  // run your CUDA kernel(s) here
 
  // for(long i=0;i<inputArgument;i++) {
  //   printf("%ld\n",numbers[i]);
  // }

    // synchronize/finalize your CUDA computations

  // gettimeofday(&ins__tstop, NULL);
  // ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  if (cudaSuccess!=cudaFree(gpu_matrix1))
    errorexit("Error when deallocating space on the GPU");
  // if (cudaSuccess!=cudaFree(gpu_matrix2))
  //   errorexit("Error when deallocating space on the GPU");

  FreeMatrix(matrix1);
  FreeMatrix(matrix2);
}
