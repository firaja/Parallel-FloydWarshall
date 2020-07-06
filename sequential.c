#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config.h"

void showDistances(int matrix[], int n);
void populateMatrix(int* matrix, int n, int density);
void floydWarshall(int* matrix, int n);

int main(int argc, char** argv) 
{
	uint n, density;
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
	
	int* matrix;

	matrix = (int*) malloc(n * n *sizeof(int));

	populateMatrix(matrix, n, density);
				
	printf("*** Adjacency matrix:\n");
	showDistances(matrix, n);	

	struct timespec start, end;
    long long accum;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	
	floydWarshall(matrix, n);
				
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	accum = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

	printf("*** The solution is:\n");
	showDistances(matrix, n);

	printf("[SEQUENTIAL] Total elapsed time %lld ns\n", accum);	
	free(matrix);
	return 0;
}

void floydWarshall(int* matrix, int n)
{
	uint i, j, k;
	for(k = 0; k < n; k++)
	{
		for(i = 0; i < n; i++)
		{
			for(j = 0; j < n; j++)
			{
				int newPath = matrix[i * n + k] + matrix[k * n + j];
                if (matrix[i * n + j] > newPath)
                {
                    matrix[i * n + j] = newPath;
                }
				
			}
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