#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PRECISION 100000

struct Matrix* CreateMatrix(int height,int width){
  struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
  matrix->height = height;
  matrix->width = width;

  matrix->mat = (float*)malloc(height* width * sizeof(float));
  // for(int i = 0; i < height; i++){
  //   matrix->mat[i] = (float*)malloc(width * sizeof(float));
  // }
  return matrix;
}

void FreeMatrix(struct Matrix* matrix){
  // for(int i = 0; i < matrix->height; i++){
  //   free(matrix->mat[i]);
  // }
  free(matrix->mat);
  free(matrix);
}

struct Matrix* FillMatrix(struct Matrix* matrix){
  for(int i = 0; i < matrix->height; i++){
    for(int j = 0; j < matrix->width; j++){
      matrix->mat[i*matrix->width+j] = floor(((float)rand()/(float)(RAND_MAX)) * 10 * PRECISION) /PRECISION ;
    }
  }
  return matrix;
}

struct Matrix* TransposeMatrix(struct Matrix* matrix){
  struct Matrix* tmatrix = CreateMatrix(matrix->width,matrix->height);
  for(int i = 0; i < tmatrix->height; i++){
    for(int j = 0; j < tmatrix->width; j++){
      tmatrix->mat[i*tmatrix->width+j] = matrix->mat[j*matrix->width+i];
    }
  }
  return tmatrix;
}

int CompareMatrix(struct Matrix* m1,struct Matrix* m2){
  if((m1->height != m2->height) || (m1->width != m2->width)){
    printf("%i,%i vs %i,%i - sizes do not match\n",m1->height, m2->height, m1->width, m2->width);
    return 0;
  }

  for(int i = 0; i < m1->height; i++){
    for(int j = 0; j < m1->width; j++){
      if(m1->mat[i * m1->width + j] != m2->mat[i * m2->width + j]){
        return 0;
      }
    }
  }
  return 1;
}

int CompareMatrixFindidx(struct Matrix* m1,struct Matrix* m2){
  if((m1->height != m2->height) || (m1->width != m2->width)){
    printf("%i,%i vs %i,%i - sizes do not match\n",m1->height, m2->height, m1->width, m2->width);
    return -2;
  }

  for(int i = 0; i < m1->height; i++){
    for(int j = 0; j < m1->width; j++){
      if(m1->mat[i * m1->width + j] != m2->mat[i * m2->width + j]){
        return i * m1->width + j;
      }
    }
  }
  return -1;
}

int CompareMatrixFindidxAll(struct Matrix* m1,struct Matrix* m2, int start){
  if((m1->height != m2->height) || (m1->width != m2->width)){
    printf("%i,%i vs %i,%i - sizes do not match\n",m1->height, m2->height, m1->width, m2->width);
    return -2;
  }


  for(int id = start; id < m1->height*m1->width; id++){
      if(m1->mat[id] != m2->mat[id]){
          printf("%f != %f at %d\n", m1->mat[id],m2->mat[id],id);
          return CompareMatrixFindidxAll(m1,m2,id+1);
      }
    }
  return -1;
}



void PrintMatrix(struct Matrix* matrix){
  for(int j = 0; j < matrix->width; j++){
    printf("\t%i\t",j);
  }  
  printf("\n");
  for(int i = 0; i < matrix->height; i++){
      printf("%i\t",i);
      for(int j = 0; j < matrix->width; j++){
        printf("%f\t",matrix->mat[i * matrix->width+j]);
      }
      printf("\n");
  }
  printf("\n");
}


void PrintMatrixPart(struct Matrix* matrix){
  int w = 10;
  int h = 10;
  if (matrix->height< h && matrix->width<w){
    PrintMatrix(matrix);
    return;
  }

  int wStride = matrix->width/w;
  int hStride = matrix->height/h;
  for(int j = 0; j < w; j++){
    printf("\t%i\t",j);
  }  
  printf("\n");
  for(int i = 0; i < h; i++){
      printf("%i\t",i);
      for(int j = 0; j < w; j++){
        printf("%f\t",matrix->mat[i * matrix->width*hStride+j*wStride]);
      }
      printf("\n");
  }
  printf("\n");
}
