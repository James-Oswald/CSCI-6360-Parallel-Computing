
#include<mpi.h>
#include<stdio.h>
#include<stdint.h>


const char* fileName = "/mnt/d/program/CSCI-6360-Parallel-Computing/Assignment5/test.txt";
int worldSize, rank;


int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File fileHandle; 
    printf("%d\n", rank);
    uint8_t val = rank;
    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_CREATE | MPI_MODE_RDWR , MPI_INFO_NULL, &fileHandle);
    MPI_File_write_at(fileHandle, rank, &val, 1, MPI_UINT8_T, MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fileHandle);
    MPI_Finalize();
}