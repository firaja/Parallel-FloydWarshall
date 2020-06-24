// g++ -fopenmp openmp.c -o openmp.out && ./openmp.out 10
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <omp.h>
#define DEFAULT 10 //number of nodes
#define PRINTABLE 1
#define MAX 100
#define INF MAX + 1
#define DENSITY 50

void showDistances(int** matrix, uint n);
void populateMatrix(int** matrix, uint n, uint density);
void floydWarshall(int** matrix, uint n);


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
	
	int** matrix;

	matrix = (int**) malloc(n * sizeof(uint*));
	for(int i = 0; i < n; i++)
	{
		matrix[i] = (int*) malloc(n * sizeof(uint));
	}

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

void floydWarshall(int** matrix, uint n)
{
	int threads = omp_get_max_threads();
	uint i, j, k;
	#pragma omp parallel num_threads(threads) private(k) shared(matrix)
	{
		
		for(k = 0; k < n; k++)
		{	
			#pragma omp for private(i, j) collapse(2)
			for(i = 0; i < n; i++)
			{
				for(j = 0; j < n; j++)
				{
					if ((matrix[i][k] * matrix[k][j] != 0) && (i != j))
					{
						if(matrix[i][j] > matrix[i][k] + matrix[k][j] || matrix[i][j] == 0)
						{
							matrix[i][j] = matrix[i][k] + matrix[k][j];
						}
					}
				}
			}
		}
	}
}

void showDistances(int** matrix, uint n) 
{
	if(PRINTABLE)
	{
		uint i, j;
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
				if(matrix[i][j] == INF)
				{
					printf("  inf");
				}
				else
				{
					printf("%5d", matrix[i][j]);
				}
			}
			printf("\n");
		}
		printf("\n");
	}
}

void populateMatrix(int** matrix, uint n, uint density)
{
	uint i, j, value;
	srand(42);

	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			if(i == j)
			{
				matrix[i][j] = 0;
			}
			else
			{
				value = 1 + rand() % MAX;
				if(value > density)
				{
					matrix[i][j] = INF;
				}
				else
				{
					matrix[i][j] = value;
				}
			}
		}
	}
}