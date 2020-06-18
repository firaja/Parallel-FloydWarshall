#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#define DEFAULT 1000 //number of nodes

void showDistances(int** dist, int n) 
{

	int i, j;
	printf("     ");
	for(i = 0; i < n; i++)
	{
		printf("N%d   ", i);
	}
	printf("\n");
	for(i = 0; i < n; i++) {
		printf("N%d", i);
		for(j = 0; j < n; j++)
		{
			printf("%5d", dist[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}


int main(int argc, char** argv) 
{
	int n;
	if(argc == 0)
	{
		n = DEFAULT;
	}
	else
	{
		n = atoi(argv[1]);
	}
	int i, j, k;
	int** dist; //array with the distances between nodes

	//Initiate the necessary memory with malloc()
	dist = (int**) malloc(n * sizeof(int*));
	for(i = 0; i < n; i++)
		dist[i] = (int*) malloc(n * sizeof(int));
	
	time_t start, end;
	//use current time
	//time(&start);
	//to generate random numbers with rand()
	srand(time(NULL));

	//Initiate the dist with random values from 0-99
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
				dist[i][j] = (rand() % 10);
			}
		}
	}
				
	//Print initial distances
	showDistances(dist, n);	

	time(&start);
	//Calculate minimum distance paths
	//Using the Floyd Warshall algorithm
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
				
	time(&end);
	//print the final distances
	showDistances(dist, n);

	printf("Total Elapsed Time %f sec\n", difftime(end, start));	
	free(dist);
	return 0;
}