#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include  <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include "cublas.h"
// Thread block size
#define BLOCK_SIZE 16

typedef enum type_version_kernel {v0,v0_bis,v1} Type_version_kernel;

void computeGold( float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
/*
#define WA (64 * BLOCK_SIZE) // Matrix A width
#define HA (64 * BLOCK_SIZE) // Matrix A height
#define WB (64 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height
*/

#  
void compute_matrixMul_C(int W);
void compute_matrixMul_cublas(int N);
void compute_matrixMul_cuda(int N,Type_version_kernel v);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);


#endif // _MATRIXMUL_H_
