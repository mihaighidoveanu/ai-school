/////////// SERIAL IMPLEMENTATION ////////

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#include "GameOfLife.h"

using namespace std;

#pragma comment (lib, "msmpi.lib")

#define NUM_ROWS 8
#define NUM_COLS 8
#define NUM_STEPS 5
#define LIFE_PROBABILITY 0.35f

enum CommunicationTag
{
	COMM_TAG_EXCHAGE_BORDER,	
};

int main(int argc, char *argv[]) 
{
	// 0. Init part, finding rank and number of processes
	//------------------------------------------------------
	int  numprocs, rank, rc;
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) 
	{
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	GameOfLife gameOfLife(NUM_ROWS, NUM_COLS);
	RectToCompute rect;
	rect.rowStart = 0;
	rect.rowEnd = NUM_ROWS - 1;
	rect.colStart = 0;
	rect.colEnd = NUM_COLS - 1;

	gameOfLife.Initialize(LIFE_PROBABILITY, rect);

	gameOfLife.DebugPrint(rect);

	for (int i = 0; i < NUM_STEPS; i++)
	{
		gameOfLife.Update(rect);
		gameOfLife.DebugPrint(rect);
	}

	MPI_Finalize();
}
