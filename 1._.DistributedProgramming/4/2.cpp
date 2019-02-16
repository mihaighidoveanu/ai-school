/////////// 2D SPLIT ////////
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#include "GameOfLife.h"

using namespace std;

enum CommunicationTag
{
	COMM_TAG_EXCHAGE_BORDER,	
};

#define NUM_ROWS 8
#define NUM_COLS 8
#define NUM_STEPS 5
#define LIFE_PROBABILITY 0.4f

#pragma comment (lib, "msmpi.lib")

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

	if (NUM_ROWS % numprocs != 0)
	{
		printf("Num procs should divide the number of rows !\n");
		return -1;
	}

	// 2D split solution 
	//-------------------------------------------------

	// Step 1: allocate, configure the table for this process block and initialize
	const int NUM_ROWS_PER_BLOCK = (NUM_ROWS / numprocs);
	GameOfLife gameOfLife(NUM_ROWS_PER_BLOCK + 2, NUM_COLS); // + 2 for ghost rows (first and last will be ghosts)

	RectToCompute rect;
	rect.rowStart = 1;	// Don't process the ghost rows
	rect.rowEnd = NUM_ROWS_PER_BLOCK;
	rect.colStart = 0;
	rect.colEnd = NUM_COLS - 1;
	gameOfLife.Initialize(LIFE_PROBABILITY, rect);

	const int ABOVE_PROC = rank - 1;
	const int BELOW_PROC = rank + 1;
	
	// Step 2: run all the steps
	for (int i = 0; i < NUM_STEPS; i++)
	{
		// Step 2_a: Send / Recv ghost rows
		bool hasGhostUPRow   = rank != 0;	// First block doesn't have a border to receive
		bool hasGhostDownRow = rank != (numprocs - 1);

		MPI_Request reqSendAbove, reqSendBelow, reqRecvAbove, reqRecvBelow;
		MPI_Request reqToWait[4]; 
		MPI_Status outStatuses[4];
		int numReqToWait = 0;

		// Send / Recv ghost row above
		if (hasGhostUPRow)
		{
			MPI_Isend(gameOfLife.mFrontTable[rect.rowStart],   sizeof(char)   * gameOfLife.mNumCols, MPI_CHAR, ABOVE_PROC, COMM_TAG_EXCHAGE_BORDER, MPI_COMM_WORLD, &reqSendAbove);
			MPI_Irecv(gameOfLife.mFrontTable[0],               sizeof(char)   * gameOfLife.mNumCols, MPI_CHAR, ABOVE_PROC, COMM_TAG_EXCHAGE_BORDER, MPI_COMM_WORLD, &reqRecvAbove);

			reqToWait[numReqToWait++] = reqSendAbove;
			reqToWait[numReqToWait++] = reqRecvAbove;
		}

		// Send / Recv ghost row below
		if (hasGhostDownRow)
		{
			MPI_Isend(gameOfLife.mFrontTable[rect.rowEnd],     sizeof(char) * gameOfLife.mNumCols, MPI_CHAR, BELOW_PROC, COMM_TAG_EXCHAGE_BORDER, MPI_COMM_WORLD, &reqSendBelow);
			MPI_Irecv(gameOfLife.mFrontTable[rect.rowEnd + 1], sizeof(char) * gameOfLife.mNumCols, MPI_CHAR, BELOW_PROC, COMM_TAG_EXCHAGE_BORDER, MPI_COMM_WORLD, &reqRecvBelow);

			reqToWait[numReqToWait++] = reqSendBelow;
			reqToWait[numReqToWait++] = reqRecvBelow;
		}

		// Wait above operations to complete
		MPI_Waitall(numReqToWait, reqToWait, outStatuses);

		// Step 2_b - We have data now, compute new generation for my part
		gameOfLife.Update(rect);

		// Step 2_c: Wait for all to finish this step before continue - needed because all blocks need to exchange ghost rows
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// TODO: do the following things below - for simplicity i just printed from each process the partial data
	// Step 3: Master should gather all partial results and create a single big GameOfLife entity
	// WARNING: You can't send MPI_send (&gameOfLife.mFrontTable[0][0] , ...) and expect to send all data !!!!!!! because data is not continuous

	printf("\n\n\nProcess %d table is:", rank);
	gameOfLife.DebugPrint(rect);

	MPI_Finalize();
	return 0;
}
