

#include<mpi.h>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include"clockcycle.h"

extern void cudaInit(int rank, size_t localArraySize);
extern void cudaReduce(const double* localArray, double* result);

#define numDoubles 1610612736
#define clockFrequency 512000000

int main(int argc, char** argv){

    //Setup
    MPI_Init(&argc, &argv);
    int worldSize, rank; 
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Initialize the array of doubles
    assert(numDoubles%worldSize == 0);
    size_t localArraySize = numDoubles/worldSize;
    double* localArray = (double*)malloc(localArraySize*sizeof(double));
    for(size_t i = 0; i < localArraySize; i++)
        localArray[i] = rank*localArraySize + i;
    MPI_Barrier(MPI_COMM_WORLD);

    
    cudaInit(); //select device and cuda and memory allocation

    uint64_t startTime = clock_now();
    double localSum = 0;
    for(int i = 0; i < localArraySize; i++)
        localSum += localArray[i];
    //global sum with MPI reduce
    double globalSum;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    uint64_t endTime = clock_now();

    //Print result
    double time_in_secs = ((double)(endTime - startTime))/clockFrequency;
    if(rank == 0)
        printf("%lf %lf\n", globalSum, time_in_secs);

    free(localArray);
    MPI_Finalize();
}