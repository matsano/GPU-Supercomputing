

#include  "matrixMul.h"



void compute_matrixMul_cublas(int N)
{
 
  float alpha = 1.0f, beta = 0.0f;

  
 
  
  // allocate host memory for matrices A and B
  unsigned int size_A = N * N;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*) malloc(mem_size_A);
  //float* h_Abis = (float*) malloc(mem_size_A);
  unsigned int size_B = N * N;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*) malloc(mem_size_B);
  //float* h_Bbis = (float*) malloc(mem_size_B);
  
  // allocate host memory for the result
  unsigned int size_C = N * N;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C = (float*) malloc(mem_size_C);
  
  
  cublasInit();

 // set seed for rand()
    srand(2006);
   // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
  
  // Allocate CUDA events that we'll use for timing
    cudaEvent_t record_event[5];
    float time_msec[4];
    for (int i=0;i<5;i++){
    	cudaEventCreate(record_event+i);   
	}

    // Record the start event
    cudaDeviceSynchronize();
    cudaEventRecord(record_event[0], NULL);
  
  float* d_A;
  cublasAlloc(N*N, sizeof(float), (void **)&d_A);
  float* d_B;
  cublasAlloc(N*N, sizeof(float), (void **)&d_B);
  float* d_C;
  cublasAlloc(N*N, sizeof(float), (void **)&d_C);
  

  cudaEventRecord(record_event[1], NULL);
    cudaEventSynchronize(record_event[1]);
  
  // copy host memory to device
  cublasSetMatrix(N,N, sizeof(float), h_A, N, d_A, N);
  cublasSetMatrix(N,N, sizeof(float), h_B, N, d_B, N);
  
   cudaEventRecord(record_event[2], NULL);
    cudaEventSynchronize(record_event[2]);
  
  
 
  cublasSgemm('n', 'n', N, N, N, alpha, d_A, N,d_B, N, beta, d_C, N);
  
   
  cudaEventRecord(record_event[3], NULL);
    cudaEventSynchronize(record_event[3]);

 


  cublasGetMatrix(N,N, sizeof(float), d_C,N, h_C, N);
  
  cudaEventRecord(record_event[4], NULL);
    cudaEventSynchronize(record_event[4]);
  


 cudaEventElapsedTime(time_msec+0, record_event[0], record_event[1]);
    cudaEventElapsedTime(time_msec+1, record_event[1], record_event[2]);
    cudaEventElapsedTime(time_msec+2, record_event[2], record_event[3]);
    cudaEventElapsedTime(time_msec+3, record_event[3], record_event[4]);
    time_msec[4]=time_msec[0]+time_msec[1]+time_msec[2]+time_msec[3];

printf("TOTAL : \t\t\t %f (ms) dont %.2f%% de gestion mémoire \n",time_msec[4],100*(time_msec[0]+time_msec[1]+time_msec[3])/time_msec[4]);
   

 
 
  
  cublasShutdown();


 // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    //free(reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}
