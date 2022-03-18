#include<mpi.h>
#include<assert.h>
#include<stdlib.h>
#include<stdio.h>
#include"clockcycle.h"

#define numDoubles 1610612736
#define clockFrequency 512000000

//Our cuda functions from gpu-reduce-cuda.cu
extern void cudaInit(int rank, size_t localArrayLength);
extern double cudaReduce(int threads, int blocks);

int main(int argc, char** argv){

    //Setup MPI
    MPI_Init(&argc, &argv);
    int worldSize, rank; 
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(numDoubles%worldSize == 0);

    //Setup cuda, select GPU, allocate memory, init array 
    cudaInit(rank, numDoubles/worldSize);

    //Run the reduction, first preforming a local reduce on the GPU with cudaReduce
    //then a MPI_Reduce to sum up all ranks
    uint64_t startTime = clock_now();
    double localSum = cudaReduce(512, (numDoubles/worldSize)/512);
    double globalSum; 
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    uint64_t endTime = clock_now();

    //Print results
    double time_in_secs = ((double)(endTime - startTime))/clockFrequency;
    if(rank == 0)
        printf("%e %lf\n", globalSum, time_in_secs);
    MPI_Finalize();
}