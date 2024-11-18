#include "matrix.h"
#include <stdlib.h>
#include <math.h>
// struct Matrix {
//   float** mat;
//   int width;
//   int height;
// };
#define PRECISION 100000


struct Matrix* CreateMatrix(int height,int width){
  struct Matrix* matrix = (struct Matrix*)malloc(sizeof(struct Matrix));
  matrix->height = height;
  matrix->width = width;

  matrix->mat = (float**)malloc(height * sizeof(float*));
  for(int i = 0; i < height; i++){
    matrix->mat[i] = (float*)malloc(width * sizeof(float));
  }
  return matrix;
}

void FreeMatrix(struct Matrix* matrix){
  for(int i = 0; i < matrix->height; i++){
    free(matrix->mat[i]);
  }
  free(matrix->mat);
  free(matrix);
}

struct Matrix* FillMatrix(struct Matrix* matrix){
  for(int i = 0; i < matrix->height; i++){
    for(int j = 0; j < matrix->width; j++){
      matrix->mat[i][j] = floor(((float)rand()/(float)(RAND_MAX)) * 10 * PRECISION) /PRECISION ;
    }
  }
  return matrix;
}
// float** FillMatrix(float** matrix, int width,int height){
//   for(int i = 0; i < height; i++){
//     for(int j = 0; j < width; j++){
//       matrix[i][j] = ((float)rand()/(float)(RAND_MAX)) * 10;
//     }
//   }
//   return matrix;
// }

struct Matrix* TransposeMatrix(struct Matrix* matrix){
  struct Matrix* tmatrix = CreateMatrix(matrix->width,matrix->height);
  for(int i = 0; i < tmatrix->height; i++){
    for(int j = 0; j < tmatrix->width; j++){
      tmatrix->mat[i][j] = matrix->mat[j][i];
    }
  }
  return tmatrix;
}

int CompareMatrix(struct Matrix* m1,struct Matrix* m2){
  if((m1->height != m2->height) || (m1->width != m2->width)){
    printf("%i,%i , %i,%i\n",m1->height, m2->height, m1->width, m2->width);
    return 0;
  }

  for(int i = 0; i < m1->height; i++){
    for(int j = 0; j < m1->width; j++){
      if(m1->mat[i][j] != m2->mat[i][j]){
        return 0;
      }
    }
  }
  return 1;
}


void PrintMatrix(struct Matrix* matrix){
  for(int j = 0; j < matrix->width; j++){
    printf("\t%i\t",j);
  }  
  printf("\n");
  for(int i = 0; i < matrix->height; i++){
      printf("%i\t",i);
      for(int j = 0; j < matrix->width; j++){
        printf("%f\t",matrix->mat[i][j]);
      }
      printf("\n");
  }
  printf("\n");
}

// void PrintMatrix(float** matrix, int w,int h){
//   for(int j = 0; j < w; j++){
//   printf("\t%i\t",j);
//   }  
//   printf("\n");
//   for(int i = 0; i < h; i++){
//     printf("%i\t",i);
//     for(int j = 0; j < w; j++){
//       printf("%f\t",matrix[i][j]);
//     }
//     printf("\n");
//   }
// }