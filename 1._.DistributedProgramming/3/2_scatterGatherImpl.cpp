#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

#pragma comment (lib, "msmpi.lib")

// Note, the function below and data structure is used all processes, duplicated in the code section of each process
//---------------------------------------------
struct Vec2D
{
	float x;
	float y;
};

float ComputeSingleDist(const Vec2D& a, const Vec2D& b)
{
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
}
//---------------------------------------------

enum CommunicationTag
{
	COMM_TAG_MASTER_SEND_TASK,
	COMM_TAG_MASTER_SEND_TERMINATE,
	COMM_TAG_SLAVE_SEND_RESULT,
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

	printf("i'm rank %d. Num procs %d\n", rank, numprocs);fflush(stdout);

	// 1. Different machine code...
	//------------------------------------------------------
	if (rank == 0)
	{
		const int desiredSize = 10000000; 
		const int sz = (desiredSize / numprocs) * numprocs; // Making sure that we create a number of elements divisible with numprocs to be easier to manage indices..
		Vec2D* arrA = new Vec2D[sz];
		Vec2D* arrB = new Vec2D[sz];
		if (!arrA || !arrB)
		{
			printf("Can't allocate so many elements. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}

		// Init data...
		for (int i = 0; i < sz; i++)
		{
			arrA[i].x = 0;
			arrA[i].y = 0;

			arrB[i].x = 1;
			arrB[i].y = 1;
		}

		// I will send equal chunks to slaves. I will also compute a chunk by myself
		assert(sz % numprocs == 0);	// It this raises we have a problem 
		int chunkSize = sz / numprocs;
		
		Vec2D* myA = (Vec2D*)malloc(chunkSize * sizeof(Vec2D));
		Vec2D* myB = (Vec2D*)malloc(chunkSize * sizeof(Vec2D));

		MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		assert(chunkSize == sz / numprocs &&" he modified the value without being told :(\n");

		printf("MASTER: I'm going to send %d Vec2D to all slaves\n", chunkSize * sizeof(Vec2D)); fflush(stdout);
		MPI_Scatter(arrA, chunkSize * sizeof(Vec2D), MPI_CHAR, myA, chunkSize * sizeof(Vec2D), MPI_CHAR, 0, MPI_COMM_WORLD);
		printf("MASTER: Between scattering\n"); fflush(stdout);
		MPI_Scatter(arrB, chunkSize * sizeof(Vec2D), MPI_CHAR, myB, chunkSize * sizeof(Vec2D), MPI_CHAR, 0, MPI_COMM_WORLD);
		/*
		int nextIndexToSend = chunkSize; // chunk 0 is for master..
		for (int i = 1; i < numprocs; i++)
		{
			MPI_Send(&arrA[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			MPI_Send(&arrB[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			nextIndexToSend += chunkSize;
		}
		*/

		// Compute my chunk 
		float localSum = 0;
		for (int i = 0; i < chunkSize; i++)
		{
			localSum += ComputeSingleDist(arrA[i], arrB[i]);
		}

		// Wait results from others
		MPI_Status(stats);
		for (int i = 1; i < numprocs; i++)
		{
			float slaveResult = 0;
			MPI_Recv(&slaveResult, 1, MPI_FLOAT, i, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &stats);

			localSum += slaveResult;
		}

		cout<<" Total Sum " << localSum <<"\n";

		// Cleanup
		delete [] arrA;
		delete [] arrB;
		free(myA);
		free(myB);
	}
	else
	{
		// Receive tasks. We use MPI_Probe because we don't know the size of the transmitted data. this is blocking until we have a message in receive queue. We could also use MPI_Iprobe which is asynchronous !
		int chunkSize = 0;
		MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		printf("Slave %d: I'm going to receive %d Vec2D from master\n", rank, chunkSize * sizeof(Vec2D)); fflush(stdout);

		Vec2D* arrA = new Vec2D[chunkSize];
		Vec2D* arrB = new Vec2D[chunkSize];
		if (!arrA || !arrB)
		{
			printf("Can't allocate so many elements. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}

		MPI_Scatter(nullptr, 0, MPI_CHAR, arrA, chunkSize * sizeof(Vec2D), MPI_CHAR, 0, MPI_COMM_WORLD);
		printf("Slave %d: Between scattering\n", rank); fflush(stdout);
		MPI_Scatter(nullptr, 0, MPI_CHAR, arrB, chunkSize * sizeof(Vec2D), MPI_CHAR, 0, MPI_COMM_WORLD);

		//MPI_Recv(arrA, sizeCount, MPI_CHAR, MPI_ANY_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &stats); 
		//MPI_Recv(arrB, sizeCount, MPI_CHAR, MPI_ANY_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &stats);

		// Compute my chunk and send back results
		float localSum = 0;
		for (int i = 0; i < chunkSize; i++)
		{
			localSum += ComputeSingleDist(arrA[i], arrB[i]);			
		}

		MPI_Send(&localSum, 1, MPI_FLOAT, 0, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);

		// Cleanup
		delete [] arrA;
		delete [] arrB;
	}

	MPI_Finalize();
}
