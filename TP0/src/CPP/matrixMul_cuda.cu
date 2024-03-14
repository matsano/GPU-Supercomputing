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
// includes, kernels
#include "matrixMul_kernel.cuh"


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void compute_matrixMul_cuda(int N,Type_version_kernel version_kernel)
{
   
    // set seed for rand()
    //srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = N * N;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = N * N;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // allocate host memory for the result
    unsigned int size_C = N * N;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C = (float*) malloc(mem_size_C);
    
 

 
 // Allocate CUDA events that we'll use for timing
    cudaEvent_t record_event[5];
    float time_msec[4];
    for (int i=0;i<5;i++){
    	cudaEventCreate(record_event+i);   
	}

    // Record the start event
    cudaDeviceSynchronize();
    cudaEventRecord(record_event[0], NULL);

    // allocate device memory
    float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);

    // allocate device memory for result
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    cudaEventRecord(record_event[1], NULL);
    cudaEventSynchronize(record_event[1]);

   
   
 
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) ;
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice) ;

  
   cudaEventRecord(record_event[2], NULL);
    cudaEventSynchronize(record_event[2]);

  

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N/threads.x, N/threads.y);

    // execute the kernel
    switch(version_kernel){
    case v0 :  matrixMul_v0<<< grid, threads >>>(d_C, d_A, d_B, N); break;
    case v0_bis :  matrixMul_v0_bis<<< grid, threads >>>(d_C, d_A, d_B, N); break;
    case v1 :  matrixMul_v1<<< grid, threads >>>(d_C, d_A, d_B, N,N); break;
    }


    cudaEventRecord(record_event[3], NULL);
    cudaEventSynchronize(record_event[3]);
   

    // copy result from device to host
     cudaMemcpy(h_C, d_C, mem_size_C,
	cudaMemcpyDeviceToHost) ;

  cudaEventRecord(record_event[4], NULL);
    cudaEventSynchronize(record_event[4]);



    cudaEventElapsedTime(time_msec+0, record_event[0], record_event[1]);
    cudaEventElapsedTime(time_msec+1, record_event[1], record_event[2]);
    cudaEventElapsedTime(time_msec+2, record_event[2], record_event[3]);
    cudaEventElapsedTime(time_msec+3, record_event[3], record_event[4]);
    time_msec[4]=time_msec[0]+time_msec[1]+time_msec[2]+time_msec[3];
    printf("TOTAL : \t\t\t %f (ms) dont %.2f%% de gestion mémoire \n",time_msec[4],100*(time_msec[0]+time_msec[1]+time_msec[3])/time_msec[4]);
   
    

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

