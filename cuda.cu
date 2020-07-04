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
	

	for(int k = 0; k < n; k++)
	{
		floydWarshallKernel<<<dim3(iDivUp(n, threadsPerBlock), n), threadsPerBlock>>>(k, deviceMatrix, n);
	}
	cudaDeviceSynchronize();

	cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);

	cudaFree(deviceMatrix);

	cudaError err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	}
}

__global__ void floydWarshallKernel(int k, int *matrix, int n)
{
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < n && x < n) 
	{
		int newPath = matrix[y * n + k] + matrix[k * n + x];
		int oldPath = matrix[y * n + x];
		if (oldPath > newPath)
		{
			matrix[y * n + x] = newPath;		
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
	return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}