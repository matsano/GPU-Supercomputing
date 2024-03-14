#include "mylib.h"
#include "mylib.cuh"


__global__ void kernel_seuillageGPU(unsigned char *d_image_in, unsigned char *d_image_out,int size_j)
{
	float Csum;
	int i, j, k, iFirst, jFirst;

	iFirst = blockIdx.x*BLOCK_SIZE; // num de block dans la grille de block
	jFirst = blockIdx.y*BLOCK_SIZE;

	i = iFirst + threadIdx.x;// recuperer l'identifiant d'un thread dans les blocs
	j = jFirst + threadIdx.y;

	float nr = 0;

nr=d_image_in[2+j*3+i*3*size_j]/sqrtf(d_image_in[0+j*3+i*3*size_j]*d_image_in[0+j*3+i*3*size_j]+d_image_in[1+j*3+i*3*size_j]*d_image_in[1+j*3+i*3*size_j]+d_image_in[2+j*3+i*3*size_j]*d_image_in[2+j*3+i*3*size_j]);

	if(nr > 0.7)
		d_image_out[1+j*3+i*3*size_j] = d_image_in[2+j*3+i*3*size_j];
	else
		d_image_out[1+j*3+i*3*size_j] = d_image_in[1+j*3+i*3*size_j]; 

	d_image_out[0+j*3+i*3*size_j] = d_image_in[0+j*3+i*3*size_j];
	d_image_out[2+j*3+i*3*size_j] = d_image_in[2+j*3+i*3*size_j];


}

__global__ void kernel_toGreyGPU(unsigned char *d_image_in, unsigned char *d_image_out,int size_j)
{
	int i, j, k, iFirst, jFirst;

	iFirst = blockIdx.x*BLOCK_SIZE; // num de block dans la grille de block
	jFirst = blockIdx.y*BLOCK_SIZE;

	i = iFirst + threadIdx.x;// recuperer l'identifiant d'un thread dans les blocs
	j = jFirst + threadIdx.y;

	d_image_out[j+i*size_j] = (d_image_in[0+j*3+i*3*size_j]+d_image_in[1+j*3+i*3*size_j]+d_image_in[2+j*3+i*3*size_j])/3;

}

__global__ void kernel_sobelGPU(unsigned char *d_image_in, unsigned char *d_image_out,int size_j, int nthreadsx, int nthreadsy)
{
	int i, j, k, iFirst, jFirst;

	iFirst = blockIdx.x*BLOCK_SIZE; // num de block dans la grille de block
	jFirst = blockIdx.y*BLOCK_SIZE;

	i = iFirst + threadIdx.x;// recuperer l'identifiant d'un thread dans les blocs
	j = jFirst + threadIdx.y;

	int dx, dy, grad;

	// on the edges
	//    left most blocks
	// if   (((blockIdx.x % BLOCK_SIZE == 0) && threadIdx.x == 0)
	// //   right most blocks
	// 	||((blockIdx.x % BLOCK_SIZE == (BLOCK_SIZE-1)) && threadIdx.x == (nthreadsx-1))
	// //      up most blocks
	// 	||((blockIdx.y % BLOCK_SIZE == 0) && threadIdx.y == 0)
	// //    down most blocks
	// 	||((blockIdx.y % BLOCK_SIZE == (BLOCK_SIZE-1)) && threadIdx.y == (nthreadsy-1)))
	// {
	// 	dx=0;
	// 	dy=0;
	// // not on the edges
	// }else{
		dx = (-1)*d_image_in[(j-1)+(i-1)*size_j] + ( 0)*d_image_in[(j-1)+(i)*size_j] + ( 1)*d_image_in[(j-1)+(i+1)*size_j] +
			 (-2)*d_image_in[(j)+(i-1)*size_j]   + ( 0)*d_image_in[(j)+(i)*size_j]   + ( 2)*d_image_in[(j)+(i+1)*size_j]   +
			 (-1)*d_image_in[(j+1)+(i-1)*size_j] + ( 0)*d_image_in[(j+1)+(i)*size_j] + ( 1)*d_image_in[(j+1)+(i+1)*size_j];
		dy = (-1)*d_image_in[(j-1)+(i-1)*size_j] + (-2)*d_image_in[(j-1)+(i)*size_j] + (-1)*d_image_in[(j-1)+(i+1)*size_j] +
			 ( 0)*d_image_in[(j)+(i-1)*size_j]   + ( 0)*d_image_in[(j)+(i)*size_j]   + ( 0)*d_image_in[(j)+(i+1)*size_j]   +
			 ( 1)*d_image_in[(j+1)+(i-1)*size_j] + ( 2)*d_image_in[(j+1)+(i)*size_j] + ( 1)*d_image_in[(j+1)+(i+1)*size_j];
	//}

	grad = sqrtf(dx*dx+dy*dy);

	d_image_out[j+i*size_j] = (char)grad;

}


Mat seuillageGPU( Mat in)
{
	cudaError_t error;
	Mat out;
	out.create(in.rows,in.cols,CV_8UC3);
	
	// allocate host memory
	unsigned char *h_image_in_GPU ;
	h_image_in_GPU=in.data;
	
	/*cudaEvent_t start,stop,start_mem,stop_mem;
	error = cudaEventCreate(&start_mem);
	error = cudaEventCreate(&stop_mem);
	
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);*/
	
	// images on device memoryÍÍÍ
	unsigned char *d_image_in_GPU;
	unsigned char *d_image_out_GPU;
	
	const unsigned long int mem_size=in.cols*in.rows*3*sizeof(unsigned char);
	
	// Alocation mémoire de d_image_in et d_image_out sur la carte GPU
	cudaMalloc((void**) &d_image_in_GPU,mem_size );
	cudaMalloc((void**) &d_image_out_GPU, mem_size);
	
	// copy host memory to device
	cudaMemcpy(d_image_in_GPU, h_image_in_GPU,mem_size ,cudaMemcpyHostToDevice);
	
	//error = cudaEventRecord(stop_mem, NULL);
	
	// Wait for the stop event to complete
	//error = cudaEventSynchronize(stop_mem);
	//float msecMem = 0.0f;
	//error = cudaEventElapsedTime(&msecMem, start, stop_mem);
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(in.rows/BLOCK_SIZE,in.cols/BLOCK_SIZE);
	
	// lancement des threads executé sur la carte GPU
	kernel_seuillageGPU<<< grid, threads >>>(d_image_in_GPU, d_image_out_GPU,in.cols);
	
	// Record the start event
	//error = cudaEventRecord(start_mem, NULL);
	//error = cudaEventSynchronize(start_mem);
	
	// copy result from device to host
	cudaMemcpy(out.data, d_image_out_GPU, mem_size,cudaMemcpyDeviceToHost);
	cudaFree(d_image_in_GPU);
	cudaFree(d_image_out_GPU);
	/*
	float msecTotal,msecMem2;
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	error = cudaEventElapsedTime(&msecMem2, start_mem, stop);
	*/
	return out;
}	
	// setup execution parameters -> découpage en threads


Mat sobelGPU( Mat in)
{
	cudaError_t error;
	Mat out;
	out.create(in.rows,in.cols,CV_8UC1);
	
	// allocate host memory
	unsigned char *h_image_in_GPU ;
	h_image_in_GPU=in.data;
	
	/*cudaEvent_t start,stop,start_mem,stop_mem;
	error = cudaEventCreate(&start_mem);
	error = cudaEventCreate(&stop_mem);
	
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);*/
	
	// images on device memoryÍÍÍ
	unsigned char *d_image_in_GPU;
	unsigned char *d_image_grey_GPU;
	unsigned char *d_image_out_GPU;
	
	const unsigned long int mem_size=in.cols*in.rows*sizeof(unsigned char);
	
	// Alocation mémoire de d_image_in et d_image_out sur la carte GPU
	cudaMalloc((void**) &d_image_in_GPU, 3*mem_size);
	cudaMalloc((void**) &d_image_grey_GPU, mem_size);
	cudaMalloc((void**) &d_image_out_GPU, mem_size);
	
	// copy host memory to device
	cudaMemcpy(d_image_in_GPU, h_image_in_GPU,3*mem_size ,cudaMemcpyHostToDevice);
	
	//error = cudaEventRecord(stop_mem, NULL);
	
	// Wait for the stop event to complete
	//error = cudaEventSynchronize(stop_mem);
	//float msecMem = 0.0f;
	//error = cudaEventElapsedTime(&msecMem, start, stop_mem);
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(in.rows/BLOCK_SIZE,in.cols/BLOCK_SIZE);
	
	// lancement des threads executé sur la carte GPU
	kernel_toGreyGPU<<< grid, threads >>>(d_image_in_GPU, d_image_grey_GPU,in.cols);
	
	kernel_sobelGPU<<< grid, threads >>>(d_image_grey_GPU, d_image_out_GPU,in.cols, grid.x, grid.y);
	
	// Record the start event
	//error = cudaEventRecord(start_mem, NULL);
	//error = cudaEventSynchronize(start_mem);
	
	// copy result from device to host
	cudaMemcpy(out.data, d_image_out_GPU, mem_size,cudaMemcpyDeviceToHost);
	cudaFree(d_image_in_GPU);
	cudaFree(d_image_grey_GPU);
	cudaFree(d_image_out_GPU);
	/*
	float msecTotal,msecMem2;
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	error = cudaEventElapsedTime(&msecMem2, start_mem, stop);
	*/
	return out;
}	
	// setup execution parameters -> découpage en threads


Mat nbGPU( Mat in)
{
	cudaError_t error;
	Mat out;
	out.create(in.rows,in.cols,CV_8UC1);
	
	// allocate host memory
	unsigned char *h_image_in_GPU ;
	h_image_in_GPU=in.data;
	
	/*cudaEvent_t start,stop,start_mem,stop_mem;
	error = cudaEventCreate(&start_mem);
	error = cudaEventCreate(&stop_mem);
	
	error = cudaEventRecord(start, NULL);
	error = cudaEventSynchronize(start);*/
	
	// images on device memoryÍÍÍ
	unsigned char *d_image_in_GPU;
	unsigned char *d_image_out_GPU;
	
	const unsigned long int mem_size=in.cols*in.rows*sizeof(unsigned char);
	
	// Alocation mémoire de d_image_in et d_image_out sur la carte GPU
	cudaMalloc((void**) &d_image_in_GPU, 3*mem_size);
	cudaMalloc((void**) &d_image_out_GPU, mem_size);
	
	// copy host memory to device
	cudaMemcpy(d_image_in_GPU, h_image_in_GPU,mem_size*3 ,cudaMemcpyHostToDevice);
	
	//error = cudaEventRecord(stop_mem, NULL);
	
	// Wait for the stop event to complete
	//error = cudaEventSynchronize(stop_mem);
	//float msecMem = 0.0f;
	//error = cudaEventElapsedTime(&msecMem, start, stop_mem);
	dim3 threads(BLOCK_SIZE,BLOCK_SIZE);
	dim3 grid(in.rows/BLOCK_SIZE,in.cols/BLOCK_SIZE);
	
	// lancement des threads executé sur la carte GPU
	kernel_toGreyGPU<<< grid, threads >>>(d_image_in_GPU, d_image_out_GPU,in.cols);
	
	// Record the start event
	//error = cudaEventRecord(start_mem, NULL);
	//error = cudaEventSynchronize(start_mem);
	
	// copy result from device to host
	cudaMemcpy(out.data, d_image_out_GPU, mem_size,cudaMemcpyDeviceToHost);
	cudaFree(d_image_in_GPU);
	cudaFree(d_image_out_GPU);
	/*
	float msecTotal,msecMem2;
	error = cudaEventRecord(stop, NULL);
	error = cudaEventSynchronize(stop);
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	error = cudaEventElapsedTime(&msecMem2, start_mem, stop);
	*/
	return out;
}	
	// setup execution parameters -> découpage en threads


