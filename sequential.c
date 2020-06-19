#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#define DEFAULT 1000 //number of nodes
#define PRINTABLE 0
#define INFINITY 1000000

void showDistances(int** dist, uint n);
void populateMatrix(int** dist, uint n);



int main(int argc, char** argv) 
{
	uint n;
	if(argc <= 1)
	{
		n = DEFAULT;
	}
	else
	{
		n = atoi(argv[1]);
	}
	
	uint i, j, k;
	int** dist;

	dist = (int**) malloc(n * sizeof(uint*));
	for(int i = 0; i < n; i++)
	{
		dist[i] = (int*) malloc(n * sizeof(uint));
	}

	populateMatrix(dist, n);
				
	printf("*** Adjacency matrix:\n");
	showDistances(dist, n);	

	struct timespec start, end;
    long long accum;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	
	for(k = 0; k < n; k++)
	{
		for(i = 0; i < n; i++)
		{
			for(j = 0; j < n; j++)
			{
				if ((dist[i][k] * dist[k][j] != 0) && (i != j))
				{
					if(dist[i][j] > dist[i][k] + dist[k][j] || dist[i][j] == 0)
					{
						dist[i][j] = dist[i][k] + dist[k][j];
					}
				}
			}
		}
	}
				
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	accum = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;

	printf("*** The solution is:\n");
	showDistances(dist, n);

	printf("[SEQUENTIAL] Total elapsed time %lld ns\n", accum);	
	free(dist);
	return 0;
}

void showDistances(int** dist, uint n) 
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
				printf("%5d", dist[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void populateMatrix(int** dist, uint n)
{
	uint i, j;
	srand(42);

	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			if(i == j)
			{
				dist[i][j] = 0;
			}
			else
			{
				dist[i][j] = 1 + rand() % 100;
			}
		}
	}
}