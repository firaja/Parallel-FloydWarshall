// mpicc -g -Wall mpi.c -o mpi.out && mpirun -np 5 mpi.out 20
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#define PRINTABLE 1
#define DEFAULT 1000
#define INFINITY 1000000
#define ROOT 0

void showDistances(int matrix[], int n);
void populateMatrix(int matrix[], int n, int rank, int processes);
void gatherResult(int matrix[], int n, int rank, int processes);
void castKRow(int matrix[], int n, int section, int kRow[], int k, int rank);
void floydWarshall(int matrix[], int n, int rank, int processes);


int main(int argc, char* argv[]) 
{
	
	struct timespec start, end;
    long long accum;


	int  n;
	int* matrix;
	
	int processes, rank;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == ROOT) 
	{
		if(argc <= 1)
		{
			n = DEFAULT;
		}
		else
		{
			n = atoi(argv[1]);
		}
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	matrix = malloc(n * n / processes * sizeof(int));

	populateMatrix(matrix, n, rank, processes);

	
	if (rank == ROOT) 
	{
    	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	}

	floydWarshall(matrix, n, rank, processes);

	gatherResult(matrix, n, rank, processes);

	free(matrix);
	
	MPI_Finalize();

	if (rank == ROOT) 
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		accum = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
		printf("[DISTRIBUTED] Total elapsed time %lld ns\n", accum);
	}
	return 0;
}  



void populateMatrix(int matrix[], int n, int rank, int processes) { 
	int i, j;
	int* temp_mat = NULL;

	if (rank == ROOT) {
		srand(42);
		temp_mat = malloc(n*n*sizeof(int));

		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++){
				if(i == j)
				{
					temp_mat[i*n+j] = 0;
				}
				else
				{
					temp_mat[i*n+j] = 1+ rand() % 100;
				}

			}
		}

		int split = n * n/processes;
		MPI_Scatter(temp_mat, split, MPI_INT, matrix, split, MPI_INT, 0, MPI_COMM_WORLD);

		printf("*** Adjacency matrix:\n");
		showDistances(temp_mat, n);
		free(temp_mat);
	} 
	else 
	{
		int split = n * n/processes;
		MPI_Scatter(temp_mat, split, MPI_INT, matrix, split, MPI_INT, 0, MPI_COMM_WORLD);
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
				printf("%5d", matrix[i*n+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}  

void gatherResult(int matrix[], int n, int rank, int processes) 
{
	int* temp_mat = NULL;

	if (rank == ROOT) {
		temp_mat = malloc(n * n * sizeof(int));
		int split = n * n / processes;
		MPI_Gather(matrix, split, MPI_INT, temp_mat, split, MPI_INT, 0, MPI_COMM_WORLD);

		printf("The solution is:\n");
		showDistances(temp_mat, n);

		free(temp_mat);
	} 
	else 
	{
		int split = n * n / processes;
		MPI_Gather(matrix, split, MPI_INT, temp_mat, split, MPI_INT, 0, MPI_COMM_WORLD);
	}
}



void floydWarshall(int matrix[], int n, int rank, int processes) 
{
	int k, i, j, temp;
	int* kRow = malloc(n*sizeof(int));
	int section = n / processes;

	for (k = 0; k < n; k++) 
	{
		
		castKRow(matrix, n, section, kRow, k, rank);
		
		for (i = 0; i < section; i++)
		{
			for (j = 0; j < n; j++) 
			{
				temp = matrix[i * n + k] + kRow[j];
				if (temp < matrix[i * n + j])
				{
					matrix[i * n + j] = temp;
				}
			}
		}
	}
	free(kRow);
}  


void castKRow(int matrix[], int n, int section, int kRow[], int k, int rank) 
{
	int j;
	int localK = k % section;
	int competenceHead;

	competenceHead = k / section;
	if (rank == competenceHead)
	{
		for (j = 0; j < n; j++)
		{
			kRow[j] = matrix[localK * n + j];
		}
	}
	MPI_Bcast(kRow, n, MPI_INT, competenceHead, MPI_COMM_WORLD);
}  