// nvcc cuda.cu -o cuda.out -gencode=arch=compute_75,code=compute_75
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cuda.h>
#include <ctime>
#include "config.h"

#define BLOCK_SIZE 128



__global__ void wakeGPU(int reps);
__global__ void floydWarshallKernel(int k, int *G, int N);

void floydWarshall(int *matrix, const int N, int bsize);
void populateMatrix(int *matrix, int n, int density);
void showDistances(int matrix[], int n);



int main(int argc, char* argv[])
{
	int n, density, bsize;

	if(argc <= 3)
	{
		n = DEFAULT;
		density = 100;
		bsize = BLOCK_SIZE;
	}
	else
	{
		n = atoi(argv[1]);
		density = atoi(argv[2]);
		bsize = atoi(argv[3]);
	}

	
	const int size = n * n * sizeof(int);

	printf("%d %d %d", n, density, bsize);
		
	int* matrix = (int *) malloc(size);

	populateMatrix(matrix, n, density);

	printf("*** Adjacency matrix:\n");
	showDistances(matrix, n);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	wakeGPU<<<1, bsize>>>(32);

	cudaEventRecord(start);

	floydWarshall(matrix, n, bsize);

	cudaEventRecord(stop);


	cudaEventSynchronize(stop);
	float accum = 0;
	cudaEventElapsedTime(&accum, start, stop);

	printf("*** The solution is:\n");
	showDistances(matrix, n);

	printf("[GPGPU] Total elapsed time %f ms\n", accum);	
	
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

__global__ void floydWarshallKernel(int u, int *G, int n)
{
	int v1 = blockDim.y * blockIdx.y + threadIdx.y;
	int v2 = blockDim.x * blockIdx.x + threadIdx.x;

	if (v1 < n && v2 < n) 
	{
		int newPath = G[v1 * n + u] + G[u * n + v2];
		int oldPath = G[v1 * n + v2];
		if (oldPath > newPath)
		{
			G[v1 * n + v2] = newPath;		
		}
	}
}


void floydWarshall(int *matrix, const int n, int bsize)
{
	int *deviceMatrix;
	int size = n * n * sizeof(int);

	cudaMalloc((int **) &deviceMatrix, size);	
	cudaMemcpy(deviceMatrix, matrix, size, cudaMemcpyHostToDevice);
	

	dim3 dimGrid((n + bsize - 1) / bsize, n);

	cudaFuncSetCacheConfig(floydWarshallKernel, cudaFuncCachePreferL1);
	for(int k = 0; k < n; k++)
	{
		floydWarshallKernel<<<dimGrid, bsize>>>(k, deviceMatrix, n);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);

	cudaFree(deviceMatrix);
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