//////////// 2D SPLIT WITH COMMUNICATION AND COMPUTATION OVERALPPING ////////
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

	const int numPerX = 2; // 2 x 2 blocks each operated by one processor
	const int numPerY = 2;
	
	int dim_sizes[2] = {numPerX, numPerY};	
	int wrap_around[2] = { false, false };
	int reorder = 1;
	MPI_Comm grid_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &grid_comm);

	MPI_Comm_rank(grid_comm, &rank);
	int coordinates[2];
	MPI_Cart_coords(grid_comm, rank, 2, coordinates);
	
	MPI_Comm_size(grid_comm, &numprocs);
	if (NUM_ROWS % numprocs != 0)
	{
		printf("Num procs should divide the number of rows !\n");
		MPI_Finalize();
		return -1;
	}

	// 2D split solution 
	//-------------------------------------------------

	// Step 1: allocate, configure the table for this process block and initialize
	const int NUM_ROWS_PER_BLOCK = (NUM_ROWS / numPerX);
	const int NUM_COLS_PER_BLOCK = (NUM_COLS / numPerY);
	GameOfLife gameOfLife(NUM_ROWS_PER_BLOCK + 2, NUM_COLS_PER_BLOCK + 2); // + 2 for ghost rows (first and last will be ghosts)

	RectToCompute rect;
	rect.rowStart = 1;	// Don't process the ghost rows
	rect.rowEnd = NUM_ROWS_PER_BLOCK;
	rect.colStart = 1;
	rect.colEnd = NUM_COLS_PER_BLOCK;
	gameOfLife.Initialize(LIFE_PROBABILITY, rect);

	int left, right, up, down;
	int source;
	MPI_Cart_shift(grid_comm, 0, -1, &source, &left);	MPI_Cart_shift(grid_comm, 0, +1, &source, &right);	MPI_Cart_shift(grid_comm, 1, -1, &source, &up);	MPI_Cart_shift(grid_comm, 1, +1, &source, &down);	printf("Process %d has neighb- left: %d, right: %d, up: %d, down: %d\n", left, right, up, down);
	MPI_Finalize();
	return 0;

	//printf(" ============================ INIT TABLE process: %d", rank);
	//gameOfLife.DebugPrint(rect);


	// Step 2: run all the steps
	for (int i = 0; i < NUM_STEPS; i++)
	{
		//printf("\n============================ STEP %d proc %d", i, rank);

		// Step 2_a: Send / Recv ghost rows
		bool hasGhostUPRow = rank != 0;	// First block doesn't have a border to receive
		bool hasGhostDownRow = rank != (numprocs - 1);

		MPI_Request reqSendAbove, reqSendBelow, reqRecvAbove, reqRecvBelow;
		MPI_Request reqToWait[4];
		MPI_Status outStatuses[4];
		int numReqToWait = 0;

		// Send / Recv ghost row above
		if (hasGhostUPRow)
		{
			MPI_Isend(gameOfLife.mFrontTable[rect.rowStart], sizeof(char)   * gameOfLife.mNumCols, MPI_CHAR, up, COMM_TAG_EXCHAGE_BORDER, grid_comm, &reqSendAbove);
			MPI_Irecv(gameOfLife.mFrontTable[0], sizeof(char)   * gameOfLife.mNumCols, MPI_CHAR, up, COMM_TAG_EXCHAGE_BORDER, grid_comm, &reqRecvAbove);

			reqToWait[numReqToWait++] = reqSendAbove;
			reqToWait[numReqToWait++] = reqRecvAbove;
		}

		// Send / Recv ghost row below
		if (hasGhostDownRow)
		{
			MPI_Isend(gameOfLife.mFrontTable[rect.rowEnd], sizeof(char) * gameOfLife.mNumCols, MPI_CHAR, down, COMM_TAG_EXCHAGE_BORDER, grid_comm, &reqSendBelow);
			MPI_Irecv(gameOfLife.mFrontTable[rect.rowEnd + 1], sizeof(char) * gameOfLife.mNumCols, MPI_CHAR, down, COMM_TAG_EXCHAGE_BORDER, grid_comm, &reqRecvBelow);

			reqToWait[numReqToWait++] = reqSendBelow;
			reqToWait[numReqToWait++] = reqRecvBelow;
		}

		// Step 2_b - After we issued the send/recv async operations we can update the middle part that doesn't need the upper / down ghost rows
		RectToCompute middleRect = rect;
		middleRect.rowStart++;
		middleRect.rowEnd--;
		gameOfLife.Update(middleRect);

		// Wait above operations to complete
		MPI_Waitall(numReqToWait, reqToWait, outStatuses);

		// Step 2_c - We have data now, compute the rows we left
		RectToCompute upRect = rect;
		upRect.rowEnd = upRect.rowStart;
		gameOfLife.Update(upRect);

		RectToCompute bottomRect = rect;
		bottomRect.rowStart = bottomRect.rowEnd;
		gameOfLife.Update(bottomRect);

		// Step 2_c: Wait for all to finish this step before continue - needed because all blocks need to exchange ghost rows
		MPI_Barrier(grid_comm);

		//printf("\n\n\nProcess %d table is:", rank);
		//gameOfLife.DebugPrint(rect);
	}

	// TODO: do the following things below - for simplicity i just printed from each process the partial data
	// Step 3: Master should gather all partial results and create a single big GameOfLife entity
	// WARNING: You can't send MPI_send (&gameOfLife.mFrontTable[0][0] , ...) and expect to send all data !!!!!!! because data is not continuous

	printf("\n\n\nProcess %d table is:", rank);
	gameOfLife.DebugPrint(rect);

	MPI_Finalize();
	return 0;
}
