#include "utility.h"
#include "matrix.c"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define THREAD_NUM 6
#define SEED 43526454

//64KB L1 cache/ 32KB d + 32KB i -> 8K floats
float ScalarVectorMultiply(float* v1, float* v2, int l){
  float result = 0;
  for(int i = 0;i<l;i++){
    result += v1[i] * v2[i];
  }
  return result;
}

float ScalarVectorMultiplyOffset(float* v1, float* v2, int l, int offset){
  float result = 0;
  for(int i = 0;i<l;i++){
    result += v1[i+offset] * v2[i+offset];
  }
  return result;
}


//Sequential Multiplication
long MatrixMultiply(struct Matrix* m1, struct Matrix* m2){
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
  long time = get_time(&ins__tstart, &ins__tstop);
  printf("S1 Finished: %ld microseconds\n",time);

  FreeMatrix(m3);
  FreeMatrix(tm2);

  return time;
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

//Parallel Multiplication 2
//compute every element in row sequentialy, rows computed in parallel
long MatrixMultiplyParallel2NoTranspose(struct Matrix* m1, struct Matrix* m2)
{
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel for collapse(2)
  for(int i = 0;i<m3->height;i++){
    for(int j = 0;j<m3->width;j++){
      float result = 0;
      for (int jj = 0;jj<m1->width;jj++){
        result += m1->mat[i][jj] * m2->mat[jj][j];
      }
      m3->mat[i][j] = result;
    }
  }
  gettimeofday(&ins__tstop, NULL);
  long time = get_time(&ins__tstart, &ins__tstop);
  printf("P1 Finished: %ld microseconds\n",time);

  FreeMatrix(m3);

  return time;
}


//Parallel Multiplication 2
//compute every element in row sequentialy, rows computed in parallel
long MatrixMultiplyParallel2(struct Matrix* m1, struct Matrix* m2)
{
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel for collapse(2)
  for(int i = 0;i<m3->height;i++){
    for(int j = 0;j<m3->width;j++){
      m3->mat[i][j] = ScalarVectorMultiply(m1->mat[i],tm2->mat[j],m1->width);
    }
  }
  gettimeofday(&ins__tstop, NULL);
  long time = get_time(&ins__tstart, &ins__tstop);
  printf("P2 Finished: %ld microseconds\n",time);

  FreeMatrix(m3);
  FreeMatrix(tm2);

  return time;
}

void ComputeTile(struct Matrix* m1, struct Matrix* m2, struct Matrix* m3, int tile_w, int tile_h ,int input_tile_w, int last_input_tile_w,  
int input_tile_num, int product_tile_row,int product_tile_col){
//fetch product tile to cache and set all elements to 0;
  for(int row = 0; row < tile_h;row++){
    for(int col = 0; col < tile_w;col++){
      m3->mat[row + product_tile_row * tile_h][col + product_tile_col * tile_w] = 0; 
    }
  }
  //Compute full input tiles 
  for(int input_tile = 0; input_tile < input_tile_num; input_tile++){
    //Compute parts
    for(int row = 0; row < tile_h;row++){
      for(int col = 0; col < tile_w;col++){
        m3->mat[row + product_tile_row * tile_h][col + product_tile_col * tile_w] +=  
          ScalarVectorMultiplyOffset(
            m1->mat[row + product_tile_row * tile_h],
            m2->mat[col + product_tile_col * tile_w], 
            input_tile_w, input_tile * input_tile_w);
      }
    }
  }
  //Compute last input tile
  for(int row = 0; row < tile_h;row++){
    for(int col = 0; col < tile_w;col++){
      m3->mat[row + product_tile_row * tile_h][col + product_tile_col * tile_w] +=  
        ScalarVectorMultiplyOffset(
          m1->mat[row + product_tile_row * tile_h],
          m2->mat[col + product_tile_col * tile_w], 
          last_input_tile_w, input_tile_num * input_tile_w);
    }
  }
}

//Parallel Multiplication 3
//Divide product matrix into tiles, compute each tile paralelly, dividing input matrices into smaller tiles used to compute product tiles
long MatrixMultiplyParallel3(struct Matrix* m1, struct Matrix* m2){
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  
  //Cache optimized tiles:    ProdW * ProdH + Input1W * ProdH + ProdW * Input2H = 8K floats
  // O:50x50 ; In: 135x50 
  // O: 64x64
  int product_tile_w = 80; 
  int product_tile_h = 80; // increases product tile h
  //m3->width/2;
  //m3->height/(THREAD_NUM/2);
  int input1_tile_w = 10; // limited by memory
  int input1_tile_h = product_tile_h; // = product_tile_h
  
  // int input2_tile_w = product_tile_w; // = product_tile_w
  // int input2_tile_h = 135; // limited by memory

  int input1_row_tiles = m1->width / input1_tile_w; //number of tiles in input row
  int last_input1_row_w = m1->width % input1_tile_w; //width of last tile of input row
  //int input2_column_tiles = m2->height / input2_tile_h; // = input1_row_tiles
  //int last_input2_column_h = m2->height % input2_tile_h; // = last_input1_row_w

  int tile_rows = m3->height / product_tile_h; //full tile rows
  int tile_columns = m3->width / product_tile_w; //full tile columns

  int last_tile_row_h = m3->height % product_tile_h; //last tile row height
  int last_tile_column_w = m3->width % product_tile_w; //last tile column width

  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel 
  { 
    #pragma omp for //schedule(static, 3)
    for(int prod_tile_row = 0; prod_tile_row < tile_rows; prod_tile_row+=1){
    //Compute Full ProductRowTiles
      for(int prod_tile_col = 0; prod_tile_col < tile_columns; prod_tile_col++){
      //Compute Product Tile:
        ComputeTile(m1,tm2,m3,product_tile_w,product_tile_h,input1_tile_w,last_input1_row_w,input1_row_tiles,prod_tile_row,prod_tile_col);
      }
      //Compute Last ProductRowTile 
      if(last_tile_column_w > 0)
      ComputeTile(m1,tm2,m3,last_tile_column_w,product_tile_h,last_tile_column_w,last_input1_row_w,input1_row_tiles,prod_tile_row,tile_columns);
    }
    
    //Compute Last Row of Tiles in Product Matrix
    //Compute Full ProductRowTiles
    if(last_tile_row_h > 0){
      // #pragma omp for
      for(int prod_tile_col = 0; prod_tile_col<tile_columns; prod_tile_col++){
      //Compute Product Tile:
        ComputeTile(m1,tm2,m3,product_tile_w,last_tile_row_h,input1_tile_w, last_input1_row_w,input1_row_tiles,tile_rows,prod_tile_col);
      }
      //Compute Last ProductRowTile in Last Row 
      ComputeTile(m1,tm2,m3,last_tile_column_w,last_tile_row_h,last_tile_column_w,last_input1_row_w,input1_row_tiles,tile_rows,tile_columns);
    }


  }

  gettimeofday(&ins__tstop, NULL);
  long time = get_time(&ins__tstart, &ins__tstop);
  printf("P3 Finished: %ld microseconds\n",time);

  FreeMatrix(tm2);
  FreeMatrix(m3);

  return time;
}



//Parallel Multiplication 3
//Divide product matrix into tiles, compute each tile paralelly, dividing input matrices into smaller tiles used to compute product tiles
long MatrixMultiplyParallel4(struct Matrix* m1, struct Matrix* m2, int product_tile_w, int product_tile_h, int input_tile_w){
  struct Matrix* tm2 = TransposeMatrix(m2);
  struct Matrix* m3 = CreateMatrix(m1->height,m2->width);

  struct timeval ins__tstart, ins__tstop;
  
  //Cache optimized tiles:    ProdW * ProdH + Input1W * ProdH + ProdW * Input2H = 8K floats
  // int product_tile_w = 50; 
  // int product_tile_h = 50; // increases product tile h
  // int input_tile_w = 55; // limited by memory

  int input_row_tiles = m1->width / input_tile_w; //number of tiles in input row
  int last_input_row_w = m1->width % input_tile_w; //width of last tile of input row

  int tile_rows = m3->height / product_tile_h; //full tile rows
  int tile_columns = m3->width / product_tile_w; //full tile columns

  int last_tile_row_h = m3->height % product_tile_h; //last tile row height
  int last_tile_column_w = m3->width % product_tile_w; //last tile column width

  gettimeofday(&ins__tstart, NULL);
  #pragma omp parallel 
  { 
    #pragma omp for schedule(guided)
    for(int prod_tile_row = 0; prod_tile_row <= tile_rows; prod_tile_row+=1){
      int tile_h = prod_tile_row != tile_rows ? product_tile_h : last_tile_row_h;
      for(int prod_tile_col = 0; prod_tile_col <= tile_columns; prod_tile_col++){
        int tile_w = prod_tile_col != tile_columns ? product_tile_w : last_tile_column_w;
        //Compute Product Tile:
        ComputeTile(m1,tm2,m3,tile_w,tile_h,input_tile_w,last_input_row_w,input_row_tiles,prod_tile_row,prod_tile_col);
      }
    }
  }

  gettimeofday(&ins__tstop, NULL);
  long time = get_time(&ins__tstart, &ins__tstop);
  printf("P4 Finished: %ld microseconds\n",time);

  FreeMatrix(tm2);
  FreeMatrix(m3);

  return time;
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
  FillMatrix(matrix1);
  FillMatrix(matrix2);

  long minT1 = __INT32_MAX__;
  long avgT1 = 0;
  long maxT1 = 0;


  long minT2 = __INT32_MAX__;
  long avgT2 = 0;
  long maxT2 = 0;

  long minT4 = __INT32_MAX__;
  long avgT4 = 0;
  long maxT4 = 0;

  printf("Start Computations\n");
  for(int i = 0;i<10;i++){
    long time5 = MatrixMultiply(matrix1,matrix2);
    long time1 = MatrixMultiplyParallel2NoTranspose(matrix1,matrix2);
    long time2 = MatrixMultiplyParallel2(matrix1,matrix2);
    // long time3 = MatrixMultiplyParallel3(matrix1,matrix2);
    //Optimal tile
    long time4 = MatrixMultiplyParallel4(matrix1,matrix2,80,80,10);
    // long time5 = MatrixMultiplyParallel4(matrix1,matrix2,50,50,55);
    // long time6 = MatrixMultiplyParallel4(matrix1,matrix2,20,20,190);
    // long time1 = MatrixMultiply(matrix1,matrix2);
    avgT2+=time2;
    if(time2 > maxT2) maxT2 = time2;
    if(time2 < minT2) minT2 = time2;

    avgT4+=time4;
    if(time4 > maxT4) maxT4 = time4;
    if(time4 < minT4) minT4 = time4;

    avgT1+=time1;
    if(time1 > maxT1) maxT1 = time1;
    if(time1 < minT1) minT1 = time1;

  }
  avgT1/=10;
  avgT2/=10;
  avgT4/=10;
  long uncT1 = (maxT1 - minT1)/2;
  long uncT2 = (maxT2 - minT2)/2;
  long uncT4 = (maxT4 - minT4)/2;

  printf("T1 mean: %ld microseconds, uncertainty: %ld\n",avgT1,uncT1);
  printf("T2 mean: %ld microseconds, uncertainty: %ld\n",avgT2,uncT2);
  printf("T4 mean: %ld microseconds, uncertainty: %ld\n",avgT4,uncT4);


  FreeMatrix(matrix1);
  FreeMatrix(matrix2);
}
