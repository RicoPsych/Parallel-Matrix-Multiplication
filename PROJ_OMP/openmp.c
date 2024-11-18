#include "utility.h"
#include "matrix.c"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define THREAD_NUM 12
#define SEED 43526454

float ScalarVectorMultiply(float* v1, float* v2, int l){
  float result = 0;
  for(int i = 0;i<l;i++){
    result += v1[i] * v2[i];
  }
  return result;
}

struct Matrix* FillMatrixParallelRows(struct Matrix* matrix){
  #pragma omp parallel for collapse(2)
  for(int i = 0; i < matrix->height; i++){
    for(int j = 0; j < matrix->width; j++){
      matrix->mat[i][j] = floor(((float)rand()/(float)(RAND_MAX)) * 10 * PRECISION) /PRECISION ;
    }
  }
  return matrix;
}

struct Matrix* TransposeMatrixParallelRows(struct Matrix* matrix){
  struct Matrix* tmatrix = CreateMatrix(matrix->width,matrix->height);
  #pragma omp parallel for collapse(2)
  for(int i = 0; i < tmatrix->height; i++){
    for(int j = 0; j < tmatrix->width; j++){
      tmatrix->mat[i][j] = matrix->mat[j][i];
    }
  }
  return tmatrix;
}

struct Matrix* FillMatrixParallelChunks(struct Matrix* matrix){
  int slice_h = matrix->height/(THREAD_NUM/2);
  int slice_w = matrix->width/2;

  #pragma omp parallel
  {
    int thread = omp_get_thread_num();
    // printf("ThreadOMP:%d, ThreadLoop:%d",omp_get_thread_num(),thread);
    int slice_i = (thread / 2) * slice_h; // 0 | 1 | 2 | 3
    int slice_j = (thread / (THREAD_NUM/2)) * slice_w; //  0 | 1

    for(int i = 0;i<slice_h;i++){
      for(int j = 0;j<slice_w;j++){
        matrix->mat[slice_i+i][slice_j+j] = floor(((float)rand()/(float)(RAND_MAX)) * 10 * PRECISION) /PRECISION ;
      }
    }
  }
  return matrix;
}

struct Matrix* FillMatrixParallelChunksRows(struct Matrix* matrix){
  int slice_h = matrix->height/(THREAD_NUM);
  int slice_w = matrix->width;

  #pragma omp parallel
  {
    int thread = omp_get_thread_num();
    // printf("ThreadOMP:%d, ThreadLoop:%d",omp_get_thread_num(),thread);
    int slice_i = (thread) * slice_h; // 0 | 1 | 2 | 3
    int slice_j = 0; //  0 | 1

    for(int i = 0;i<slice_h;i++){
      for(int j = 0;j<slice_w;j++){
        matrix->mat[slice_i+i][slice_j+j] = floor(((float)rand()/(float)(RAND_MAX)) * 10 * PRECISION) /PRECISION ;
      }
    }
  }
  return matrix;
}




//Sequential Multiplication
struct Matrix* MatrixMultiply(struct Matrix* m1, struct Matrix* m2){
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  for(int i = 0;i<m3->height;i++){
    for(int j = 0;j<m3->width;j++){
      m3->mat[i][j] = ScalarVectorMultiply(m1->mat[i],tm2->mat[j],m1->width);
    }
  }

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "S1");

  FreeMatrix(tm2);
  return m3;
}

//Parallel Multiplication 1
//compute every element in row in parallel, rows computed sequentially
struct Matrix* MatrixMultiplyParallel1(struct Matrix* m1, struct Matrix* m2)
{
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  printf("Start P1\n");
  gettimeofday(&ins__tstart, NULL);

  for(int i = 0;i<m3->height;i++){
    #pragma omp parallel for 
    for(int j = 0;j<m3->width;j++){
      m3->mat[i][j] = ScalarVectorMultiply(m1->mat[i],tm2->mat[j],m1->width);
    }
  }
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "P1");

  FreeMatrix(tm2);
  return m3;
}

//Parallel Multiplication 1
//compute every element in row sequentialy, rows computed in parallel
struct Matrix* MatrixMultiplyParallel2Unoptimized(struct Matrix* m1, struct Matrix* m2)
{
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  printf("Start P2 Unoptimized\n");
  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel for
  for(int i = 0;i<m3->height;i++){
    for(int j = 0;j<m3->width;j++){
      m3->mat[i][j] = ScalarVectorMultiply(m1->mat[i],tm2->mat[j],m1->width);
    }
  }
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "P2 Unoptimized");

  FreeMatrix(tm2);
  return m3;
}



//Parallel Multiplication 1
//compute every element in row sequentialy, rows computed in parallel
struct Matrix* MatrixMultiplyParallel2(struct Matrix* m1, struct Matrix* m2)
{
  // struct Matrix* tm2 = TransposeMatrix(m2);

  struct Matrix* tm2 = TransposeMatrixParallelRows(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  printf("Start P2\n");
  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel for collapse(2)
  for(int i = 0;i<m3->height;i++){
    for(int j = 0;j<m3->width;j++){
      m3->mat[i][j] = ScalarVectorMultiply(m1->mat[i],tm2->mat[j],m1->width);
    }
  }
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "P2");

  FreeMatrix(tm2);
  return m3;
}



//Parallel Multiplication 3
//Divide result matrix into n(thread num) parts, each part computed in parallel
struct Matrix* MatrixMultiplyParallel3(struct Matrix* m1, struct Matrix* m2){
  struct Matrix* tm2 = TransposeMatrixParallelRows(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;

  int slice_w = m3->width/2;
  int slice_h = m3->height/(THREAD_NUM/2);
  
  // printf("h:%d",slice_h);
  // printf("w:%d",slice_w);
  
  printf("Start P3\n");
  gettimeofday(&ins__tstart, NULL);

  #pragma omp parallel 
  {
    int thread = omp_get_thread_num();
    int slice_i = (thread / 2) * slice_h; // 0 | 1 | 2 | 3
    int slice_j = (thread / (THREAD_NUM/2)) * slice_w; //  0 | 1

    for(int i = 0;i<slice_h;i++){
      for(int j = 0;j<slice_w;j++){
        m3->mat[slice_i+i][slice_j+j] = ScalarVectorMultiply(m1->mat[slice_i+i],tm2->mat[slice_j+j], m1->width);
      }
    }
  }

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "P3");

  FreeMatrix(tm2);
  return m3;
}

//Parallel Multiplication 4
//Divide result matrix into n(thread num) parts, each part computed in parallel
struct Matrix* MatrixMultiplyParallel4(struct Matrix* m1, struct Matrix* m2){
  struct Matrix* tm2 = TransposeMatrixParallelRows(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;

  int slice_w = m3->width;
  int slice_h = m3->height/THREAD_NUM;
  
  // printf("h:%d",slice_h);
  // printf("w:%d",slice_w);
  
  printf("Start P4\n");
  gettimeofday(&ins__tstart, NULL);

  #pragma omp parallel 
  {
    int thread = omp_get_thread_num();
    int slice_i = (thread) * slice_h; // 0 | 1 | 2 | 3
    int slice_j = 0;

    for(int i = 0;i<slice_h;i++){
      for(int j = 0;j<slice_w;j++){
        m3->mat[slice_i+i][slice_j+j] = ScalarVectorMultiply(m1->mat[slice_i+i],tm2->mat[slice_j+j], m1->width);
      }
    }
  }

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, "P4");

  FreeMatrix(tm2);
  return m3;
}


int main(int argc,char **argv) {
  srand(SEED);

  int w,h;
  switch (argc)
  {
    case 1:
      w = h = 1000;
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

  // //set number of threads
  omp_set_num_threads(THREAD_NUM);
  
  struct Matrix* matrix1 = CreateMatrix(h,w);
  struct Matrix* matrix2 = CreateMatrix(w,h); 
  struct Matrix* matrix3 = CreateMatrix(h,w); 
  
  // FillMatrix(matrix1);
  // FillMatrix(matrix2);

  FillMatrixParallelRows(matrix2);
  FillMatrix(matrix3);
  
  // FillMatrixParallelChunks(matrix1);
  // FillMatrixParallelChunks(matrix2);

  // FillMatrixParallelChunksRows(matrix1);
  // FillMatrixParallelChunksRows(matrix2);


  printf("Start Computations\n");
  // struct Matrix* m3 = MatrixMultiplyParallel1(matrix1,matrix2);
  struct Matrix* m2 = MatrixMultiplyParallel2Unoptimized(matrix3,matrix2);
  
  FillMatrixParallelRows(matrix1);
  struct Matrix* m4 = MatrixMultiplyParallel2(matrix1,matrix2);
  // struct Matrix* m5 = MatrixMultiplyParallel3(matrix1,matrix2);
  // struct Matrix* m6 = MatrixMultiplyParallel4(matrix1,matrix2);

  // printf("Start Seq\n");
  // struct Matrix* m2 = MatrixMultiply(matrix1,matrix2);

  // PrintMatrix(m2);
  // PrintMatrix(m3);
  // PrintMatrix(m4);
  // printf("Mat compare 1: %i\n", CompareMatrix(m2,m3));
  // printf("Mat compare 2: %i\n", CompareMatrix(m2,m4));

  // // // synchronize/finalize your computations
  // gettimeofday(&ins__tstop, NULL);
  // ins__printtime(&ins__tstart, &ins__tstop, "");
  
  FreeMatrix(m2);
  // FreeMatrix(m3);
  FreeMatrix(m4);
  // FreeMatrix(m5);
  // FreeMatrix(m6);
  FreeMatrix(matrix1);
  FreeMatrix(matrix2);
  FreeMatrix(matrix3);

}
