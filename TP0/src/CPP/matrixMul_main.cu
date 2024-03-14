/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include "matrixMul.h"


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, const char** argv)
{
//findCudaDevice(argc,argv);

  int n;

 printf("\nmatrix_Mul_CUDA_C\n\n");
  for (n=7;n<11;n++){
    unsigned int taille_matrice;
    taille_matrice=(unsigned int)pow((float)2.0,n);
    printf("MATRICE DE TAILLE %d\n",taille_matrice);
    compute_matrixMul_C(taille_matrice);
  }


  printf("\nmatrix_Mul_CUDA_v0\n\n");
    for (n=7;n<11;n++){
      unsigned int taille_matrice;
      taille_matrice=(unsigned int)pow((float)2.0,n);
      printf("MATRICE DE TAILLE %d\n",taille_matrice);
      compute_matrixMul_cuda(taille_matrice,v0);
    }


    printf("\nmatrix_Mul_CUDA_v0_bis\n\n");
       for (n=7;n<11;n++){
         unsigned int taille_matrice;
         taille_matrice=(unsigned int)pow((float)2.0,n);
         printf("MATRICE DE TAILLE %d\n",taille_matrice);
         compute_matrixMul_cuda(taille_matrice,v0_bis);
       }


       printf("\nmatrix_Mul_CUDA_v1\n\n");
            for (n=7;n<11;n++){
              unsigned int taille_matrice;
              taille_matrice=(unsigned int)pow((float)2.0,n);
              printf("MATRICE DE TAILLE %d\n",taille_matrice);
              compute_matrixMul_cuda(taille_matrice,v1);
            }



            printf("\nmatrix_Mul_CUBLAS\n\n");
                 for (n=7;n<11;n++){
                   unsigned int taille_matrice;
                   taille_matrice=(unsigned int)pow((float)2.0,n);
                   printf("MATRICE DE TAILLE %d\n",taille_matrice);
                   compute_matrixMul_cublas(taille_matrice);
                 }

  //CUT_EXIT(argc, argv);
}
