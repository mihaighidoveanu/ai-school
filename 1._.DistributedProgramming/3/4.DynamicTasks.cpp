// Dynamic parallelism version
// Master is sending tasks when slaves are idle
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

	// 1. Different machine code...
	//------------------------------------------------------
	if (rank == 0)
	{
		const int desiredSize = 1000000; 
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

		// Imagine that we have a queue of equal tasks.
		const int chunkSize = desiredSize / 10;
		assert(desiredSize % chunkSize == 0);

		int nextIndexToSend = 0; // point to the begging of the queue
		float localSum = 0;

		// Send a task to every slave - we know that everyone is idle
		for (int i = 1; i < numprocs; i++)
		{
			MPI_Send(&arrA[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			MPI_Send(&arrB[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, i, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			nextIndexToSend += chunkSize;
		}

		//cout<<"next " << nextIndexToSend <<" " << " desiredSize " << desiredSize;
		while(true)
		{
			// Wait for a result from any process. 
			MPI_Status stats;
			float slaveResult = 0;
			MPI_Recv(&slaveResult, 1, MPI_FLOAT, MPI_ANY_SOURCE, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD, &stats);
			localSum += slaveResult;

			if (nextIndexToSend >= desiredSize)	// End of queue
			{
				break;
			}

			// If this process sends a results it means that it becomes idle
			MPI_Send(&arrA[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, stats.MPI_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			MPI_Send(&arrB[nextIndexToSend], (chunkSize * sizeof(Vec2D)), MPI_CHAR, stats.MPI_SOURCE, COMM_TAG_MASTER_SEND_TASK, MPI_COMM_WORLD);
			nextIndexToSend += chunkSize;	

		}

		// Send termination
		cout<<"Sending termination.. " << endl;
		for (int i = 1; i < numprocs; i++)
		{
			char dummy;
			MPI_Send(&dummy, 1, MPI_CHAR, i, COMM_TAG_MASTER_SEND_TERMINATE, MPI_COMM_WORLD);
		}

		cout<<" Total Sum " << localSum <<"\n";

		// Cleanup
		delete [] arrA;
		delete [] arrB;
	}
	else
	{
		while(true)
		{
			// Receive tasks. We use MPI_Probe because we don't know the size of the transmitted data. this is blocking until we have a message in receive queue. We could also use MPI_Iprobe which is asynchronous !
			int sizeCount = 0;
			MPI_Status stats;

			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stats);
			MPI_Get_count(&stats, MPI_CHAR, &sizeCount);	// Raw byte-sized received

			// Terminate !
			if (stats.MPI_TAG == COMM_TAG_MASTER_SEND_TERMINATE)
				break;

			// Note: Be aware of memory fragmentation below !! if you have too many tasks is better to know/fix the size of the arrays and avoid allocations...
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

			cout<<"Slave " << rank <<" Sending task result to master: " << localSum <<"\n";
			MPI_Send(&localSum, 1, MPI_FLOAT, 0, COMM_TAG_SLAVE_SEND_RESULT, MPI_COMM_WORLD);

			// Cleanup
			delete [] arrA;
			delete [] arrB;
		}
	}

	MPI_Finalize();
}
