#ifndef __MATRIX_H__
#define __MATRIX_H__
struct Matrix {
  float** mat;
  int width;
  int height;
};

struct Matrix* CreateMatrix(int width,int height);
void FreeMatrix(struct Matrix* matrix);
struct Matrix* FillMatrix(struct Matrix* matrix);
void PrintMatrix(struct Matrix* matrix);
struct Matrix* TransposeMatrix(struct Matrix* matrix);
int CompareMatrix(struct Matrix* m1,struct Matrix* m2);

#endif
