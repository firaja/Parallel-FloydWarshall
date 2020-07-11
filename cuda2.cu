// nvcc cuda2.cu -o cuda2.out -gencode=arch=compute_75,code=compute_75 -O3
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cuda.h>
#include <ctime>
#include "config.h"
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 4



__global__ void wakeGPU(int reps);
__global__ void floydWarshallKernel(int k, int *matrix, int n);

void floydWarshall(int *matrix, int n, int threadsPerBlock);
void populateMatrix(int *matrix, int n, int density);
void showDistances(int matrix[], int n);
int iDivUp(int a, int b);



int main(int argc, char* argv[])
{
	int n, density, threadsPerBlock;

	if(argc <= 3)
	{
		n = DEFAULT;
		density = 100;
		threadsPerBlock = BLOCK_SIZE;
	}
	else
	{
		n = atoi(argv[1]);
		density = atoi(argv[2]);
		threadsPerBlock = atoi(argv[3]);
	}

	
	int size = n * n * sizeof(int);
		
	int* matrix = (int *) malloc(size);

	populateMatrix(matrix, n, density);

	printf("*** Adjacency matrix:\n");
	showDistances(matrix, n);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	wakeGPU<<<1, threadsPerBlock>>>(32);

	cudaEventRecord(start);

	floydWarshall(matrix, n, threadsPerBlock);

	cudaEventRecord(stop);


	cudaEventSynchronize(stop);
	float accum = 0;
	cudaEventElapsedTime(&accum, start, stop);

	printf("*** The solution is:\n");
	showDistances(matrix, n);

	printf("[GPGPU] Total elapsed time %f ms\n", accum);	

	// calculate theoretical occupancy
  	int maxActiveBlocks;
  	cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 floydWarshallKernel, threadsPerBlock, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * threadsPerBlock / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         threadsPerBlock, occupancy);
	
	free(matrix);
	
	return 0;
}



__global__ void wakeGPU(int reps)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= reps)
	{
		return;
	}
}

void floydWarshall(int *matrix, const int n, int threadsPerBlock)
{
	int *deviceMatrix;
	int size = n * n * sizeof(int);

	cudaMalloc((int **) &deviceMatrix, size);	
	cudaMemcpy(deviceMatrix, matrix, size, cudaMemcpyHostToDevice);

	dim3 dimGrid((n +  threadsPerBlock - 1)/threadsPerBlock, n);
	
	cudaFuncSetCacheConfig(floydWarshallKernel, cudaFuncCachePreferL1);
	for(int k = 0; k < n; k++)
	{
		floydWarshallKernel<<<dimGrid, threadsPerBlock>>>(k, deviceMatrix, n);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);

	cudaFree(deviceMatrix);

	cudaError err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

__global__ void floydWarshallKernel(int k, int *matrix, int n)
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < n && j < n) 
	{
		int newPath = matrix[k * n + j] + matrix[i * n + k];
		int oldPath = matrix[i * n + j];
		if (oldPath > newPath)
		{
			matrix[i * n + j] = newPath;		
		}
	}
}

void showDistances(int matrix[], int n)
{
	if(PRINTABLE)
	{
		int i, j;
		printf("     ");
		for(i = 0; i < n; i++)
		{
			printf("[%d]  ", i);
		}
		printf("\n");
		for(i = 0; i < n; i++) {
			printf("[%d]", i);
			for(j = 0; j < n; j++)
			{
				if(matrix[i * n + j] == INF)
				{
					printf("  inf");
				}
				else
				{
					printf("%5d", matrix[i * n + j]);
				}
				
			}
			printf("\n");
		}
		printf("\n");
	}
}

void populateMatrix(int *matrix, int n, int density)
{
	uint i, j, value;
	srand(42);

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++){
			if(i == j)
			{
				matrix[i*n+j] = 0;
			}
			else
			{
				value = 1 + rand() % MAX;
				if(value > density)
				{
					matrix[i*n+j] = INF;
				}
				else
				{
					matrix[i*n+j] = value;
				}
			}

		}
	}
}

int iDivUp(int a, int b)
{
	int result = ceil(1.0 * a / b);
	if(result < 1)
	{
		return 1;
	}
	else
	{
		return result;
	}
}