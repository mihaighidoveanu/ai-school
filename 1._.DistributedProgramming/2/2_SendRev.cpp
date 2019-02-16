#include <mpi.h>
#include <stdio.h>
#include <assert.h>

#pragma comment (lib, "msmpi.lib")

int main(int argc, char *argv[]) 
{
	const int rc = MPI_Init(&argc, &argv);
	int tag = 0;
	if (rc != MPI_SUCCESS) 
	{
		printf("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	int  numtasks, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("There are %d procs in network and I'm rank: %d\n", numtasks, rank);

	if (numtasks < 2)
	{
		printf("Error: create 2 processes for this test\n");
		MPI_Finalize();
		return -1;
	}

	if (rank == 0)
	{
		const int otherProc = 1;
		for (int i = 0; i < 10; i++)
		{
			MPI_Send(&i, 1, MPI_INT, otherProc, tag, MPI_COMM_WORLD);
			printf("Proc %d: I've sent value %d to Proc %d\n", rank, i, otherProc);

			MPI_Status status;
			int valueReceived = 0;
			MPI_Recv(&valueReceived, 1, MPI_INT, otherProc /* can be: MPI_ANY_SOURCE */, tag, MPI_COMM_WORLD, &status);
			assert(valueReceived == i);	// Just for testing if the right value was received...

			printf("Proc %d: I've received value %d from Proc %d\n", rank, valueReceived, otherProc);
		}
	}
	else if (rank == 1)
	{
		const int otherProc = 0;
		for (int i = 0; i < 10; i++)
		{
			MPI_Status status;
			int valueReceived = 0;
			MPI_Recv(&valueReceived, 1, MPI_INT, otherProc  /* can be: MPI_ANY_SOURCE */, tag, MPI_COMM_WORLD, &status);
			assert(valueReceived == i);

			printf("Proc %d: I've received value %d from Proc %d\n", rank, valueReceived, otherProc);


			MPI_Send(&i, 1, MPI_INT, otherProc, tag, MPI_COMM_WORLD);
			printf("Proc %d: I've sent value %d to Proc %d\n", rank, i, otherProc);
		}
	}

	MPI_Finalize();
}

