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

	printf("i'm rank %d. Num procs %d\n", rank, numprocs); fflush(stdout);

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
		const int chunkSize = sz / numprocs;
		
		int nextIndexToSend = chunkSize; // chunk 0 is for master..
		for (int i = 1; i < numprocs; i++)
		{
			MPI_Send(&arrA[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			MPI_Send(&arrB[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			nextIndexToSend += chunkSize;
		}

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
	}
	else
	{
		// Receive tasks. We use MPI_Probe because we don't know the size of the transmitted data. this is blocking until we have a message in receive queue. We could also use MPI_Iprobe which is asynchronous !
		int sizeCount = 0;
		MPI_Status stats;
		MPI_Probe(MPI_ANY_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &stats);
		MPI_Get_count(&stats, MPI_CHAR, &sizeCount);	// Raw byte-sized received

		const int chunckSize = sizeCount / sizeof(Vec2D);	// Convert to number of Vec2D structs count
		Vec2D* arrA = new Vec2D[chunckSize];
		Vec2D* arrB = new Vec2D[chunckSize];
		if (!arrA || !arrB)
		{
			printf("Can't allocate so many elements. Terminating.\n");
			MPI_Abort(MPI_COMM_WORLD, rc);
		}

		MPI_Recv(arrA, sizeCount, MPI_CHAR, MPI_ANY_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &stats); 
		MPI_Recv(arrB, sizeCount, MPI_CHAR, MPI_ANY_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD, &stats);

		// Compute my chunk and send back results
		float localSum = 0;
		for (int i = 0; i < chunckSize; i++)
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
