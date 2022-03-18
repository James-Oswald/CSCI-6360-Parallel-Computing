
#include<mpi.h>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include"clockcycle.h"

#define numDoubles 1610612736
#define clockFrequency 512000000

int main(int argc, char** argv){

    //Setup MPI and find our rank
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
    MPI_Barrier(MPI_COMM_WORLD); //Wait till we're all initialized

    //Preform Local Sum for each rank
    uint64_t startTime = clock_now();
    double localSum = 0;
    for(size_t i = 0; i < localArraySize; i++)
        localSum += localArray[i];
    //global sum with MPI_Reduce
    double globalSum;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    uint64_t endTime = clock_now();

    //Print result on rank 0
    double time_in_secs = ((double)(endTime - startTime))/clockFrequency;
    if(rank == 0) 
        printf("%e %lf\n", globalSum, time_in_secs);

    free(localArray);
    MPI_Finalize();
}

