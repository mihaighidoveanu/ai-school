#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

using namespace std;

#pragma comment (lib, "msmpi.lib")

#include <mpi.h>
#define FILESIZE 1048576
#define INTS_PER_BLK 16

int main(int argc, char **argv)
{
		int *buf, rank, nprocs, nints, bufsize;
		MPI_File fh; 
		MPI_Datatype filetype;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

		bufsize = FILESIZE/nprocs;
		buf = (int *) malloc(bufsize);
		nints = bufsize/sizeof(int);

		MPI_File_open(MPI_COMM_WORLD, "text.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
		MPI_Type_vector(nints/INTS_PER_BLK, INTS_PER_BLK, INTS_PER_BLK*nprocs, MPI_INT, &filetype); // (num blocks, elements in each block, stride = number of elements between start of each block, old datatype)
		MPI_Type_commit(&filetype);
		MPI_File_set_view(fh, INTS_PER_BLK*sizeof(int)*rank, MPI_INT, filetype, "native", MPI_INFO_NULL);	// (file handler, start displacement, basic type, file view type, how data is represented in memory", info)
		MPI_File_read_all(fh, buf, nints, MPI_INT, MPI_STATUS_IGNORE);	// file handler, buffer to write, num items, item data type to write).

		MPI_Type_free(&filetype);
		free(buf);
		
		MPI_Finalize();
		return(0);
}
