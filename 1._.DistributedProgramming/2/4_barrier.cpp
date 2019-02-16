#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])  
{
	int numtasks, rank, dest, source, rc, count, tag = 1;
	char inmsg, outmsg = 'x';
	MPI_Status Stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) 
	{
		dest = 1;
		source = 1;
		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &Stat);	// Also, can use MPI_ANY_TAG !!

		outmsg++;
		MPI_Barrier(MPI_COMM_WORLD);	// Waits for all processes to get here !!!

		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &Stat);	// Also, can use MPI_ANY_TAG !!
	}

	else if (rank == 1) 
	{
		dest = 0;
		source = 0;
		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &Stat);
		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);

		outmsg++;
		MPI_Barrier(MPI_COMM_WORLD);	// Waits for all processes to get here !!!

		rc = MPI_Recv(&inmsg, 1, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &Stat);
		rc = MPI_Send(&outmsg, 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}

	printf("%c\n", (char)inmsg);

	// Syncronize all processes 

	MPI_Finalize();
}
