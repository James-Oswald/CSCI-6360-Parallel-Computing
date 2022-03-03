#include<stdlib.h>
#include<assert.h>
#include<stdio.h>
#include<mpi.h>
#include"clockcycle.h"

#define numbersToSum ((long)2<<20)
#define clock_frequency 512000000

int MPI_P2P_reduce(const void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, int root,
                    MPI_Comm comm){
    if(root != 0)           //The final result must go to MPI rank 0
        return MPI_ERR_ROOT;
    if(op != MPI_SUM)       //Our MPI_P2P_reduce only computes MPI_SUM
        return MPI_ERR_OP;
    //Compute local sum
    const long long * localBuffer = (const long long*)sendbuf;
    int rank, comSize;
    MPI_Comm_size(comm, &comSize);
    MPI_Comm_rank(comm, &rank);

    printf("test2");
    //comSize is a power of 2, MPI_P2P_reduce builds a binary tree and must have a power of 2 number of leaves
    //for the reduction scheme specified in the assignment instructions
    assert((comSize & (comSize - 1)) == 0);

    long long int localSum = 0;
    for(int i = 0; i < count; i++)
        localSum += localBuffer[i];
    MPI_Barrier(comm); //Wait untill all local sums have been computed

    //Pairwise reduce
    MPI_Request request;
    for(int i = 1; i < comSize; i *= 2){
        if(rank%i==0){          //On level n, only 2^n's will send or recive
            if(rank%(2*i)==0){  //On level, n every rank 2*(2^n) appart will recive from the rank Rank 2^n ahead
                long long int incomingSum;
                MPI_Irecv(&incomingSum, 1, datatype, rank+i, MPI_ANY_TAG, comm, &request);
                MPI_Wait(&request, MPI_STATUS_IGNORE);
                localSum += incomingSum;
            }else{
                MPI_Isend(&localSum, 1, datatype, rank-i, 0, comm, &request);
            }
        }
        MPI_Barrier(comm);
    }

    if(rank == 0)
        *(long long int*)recvbuf = localSum;
    else
        *(long long int*)recvbuf = 0;   //On all other ranks besides 0, we return nothing

    return MPI_SUCCESS;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, comSize;
    MPI_Comm_size(MPI_COMM_WORLD, &comSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t localArraySize = (numbersToSum/comSize);
    long long int* localArray = (long long int*)malloc(localArraySize*sizeof(long long int));
    for(long long int i = 0; i < localArraySize; i++)
        localArray[i] = (long long int)rank*(long long int)localArraySize + i;

    long long int result;

    //My MPI_P2P_reduce reduction
    uint64_t startTime = clock_now();
    MPI_P2P_reduce(localArray, &result, localArraySize, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    uint64_t endTime = clock_now();
    double time_in_secs = ((double)(endTime - startTime)) / clock_frequency;
    if(rank == 0)
        printf("MPI_P2P_reduce\nResult:%lld\nTime:%lf\n\n", result, time_in_secs);

    //MPI_Reduce reduction
    startTime = clock_now();
    long long int localSum = 0;  //Sum local array
    for(int i = 0; i < localArraySize; i++)
        localSum += localArray[i];
    MPI_Reduce(&localSum, &result, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); //reduce to single elm
    endTime = clock_now();
    time_in_secs = ((double)(endTime - startTime)) / clock_frequency;
    if(rank == 0)
        printf("MPI_reduce\nResult:%lld\nTime:%lf\n", result, time_in_secs);

    free(localArray);
    MPI_Finalize();
    return 0;
}