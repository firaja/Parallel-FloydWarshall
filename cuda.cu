// nvcc cuda.cu -o cuda.out -gencode=arch=compute_75,code=compute_75
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cuda.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256
#define DEFAULT 10
#define DENSITY 50
#define PRINTABLE 1
#define MAX 100
#define INF MAX + 1


__global__ void wakeGPU(int reps);
__global__ void floydWarshallKernel(int k, int *G, int N);

void floydWarshall(int *matrix, const int N);
void populateMatrix(int *matrix, int n, int density);
void showDistances(int matrix[], int n);



int main(int argc, char* argv[])
{
	int n, density;

	if(argc <= 2)
	{
		n = DEFAULT;
		density = 100;
	}
	else
	{
		n = atoi(argv[1]);
		density = atoi(argv[2]);
	}

	
	const int size = n * n * sizeof(int);
		
	int* matrix = (int *) malloc(size);

	populateMatrix(matrix, n, density);

	printf("*** Adjacency matrix:\n");
	showDistances(matrix, n);

	wakeGPU<<<1, BLOCK_SIZE>>>(32);

	floydWarshall(matrix, n);

	printf("*** The solution is:\n");
	showDistances(matrix, n);

	
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

__global__ void floydWarshallKernel(int k, int *G, int N)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= N)
	{
		return;
	}
	int idx = N * blockIdx.y + col;

	__shared__ int best;

	if(threadIdx.x == 0)
	{
		best=G[N * blockIdx.y + k];
	}

	__syncthreads();

	if(best == INFINITY)
	{
		return;
	}
	int tmp_b = G[k * N + col];
	if(tmp_b == INFINITY)
	{
		return;
	}
	int cur = best + tmp_b;
	if(cur < G[idx])
	{
		G[idx] = cur;
	}
}


void floydWarshall(int *matrix, const int n)
{
	int *deviceMatrix;
	int size = n * n * sizeof(int);

	cudaError_t err = cudaMalloc((int **) &deviceMatrix, size);
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__ ,__LINE__);
	}
	

	err=cudaMemcpy(deviceMatrix, matrix, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	}
	

	dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, n);

	for(int k = 0; k < n; k++)
	{
		floydWarshallKernel<<<dimGrid, BLOCK_SIZE>>>(k, deviceMatrix, n);
		err = cudaDeviceSynchronize();
		if(err != cudaSuccess)
		{
			printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		}
	}

	err = cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
	}

	err = cudaFree(deviceMatrix);
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__,__LINE__);
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