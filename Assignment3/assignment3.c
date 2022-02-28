
#include<assert.h>
#include<stdio.h>
#include<mpi.h>
#include"clockcycle.h"

int MPI_P2P_reduce(const void *sendbuf, void *recvbuf, int count,
                    MPI_Datatype datatype, MPI_Op op, int root,
                    MPI_Comm comm){
    if(root != 0)           //The final result must go to MPI rank 0
        return MPI_ERR_ROOT;
    if(op != MPI_SUM)       //Our MPI_P2P_reduce only computes MPI_SUM
        return MPI_ERR_OP;
    //Compute local sum
    const int* localBuffer = (const int*)sendbuf; 
    int rank, comSize;
    MPI_Comm_size(comm, &comSize);
    MPI_Comm_rank(comm, &rank);
    int64_t localSum = 0;
    for(int i = 0; i < count; i++)
        localSum += localBuffer[i];
    
    printf("%d:%lu\n", rank, localSum);
    MPI_Barrier(comm); //Wait untill all local sums have been computed

    //Pairwise reduce
    MPI_Request request;
    for(int i = 1; i < comSize; i *= 2){
        if(rank%i==0){
            if(rank%(2*i)==0){
                uint64_t incomingSum;
                printf("%d: %d, %d \n", rank, rank%(2*i), rank+i);
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
        *(uint64_t*)recvbuf = localSum;
    else
        *(uint64_t*)recvbuf = 0;   //On all other ranks besides 0, we return nothing
    
    return MPI_SUCCESS;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, comSize;
    MPI_Comm_size(MPI_COMM_WORLD, &comSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert((comSize & (comSize - 1)) == 0); //comSize is a power of 2, MPI_P2P_reduce wont work otherwise
    
    const int sizeToSum = 2<<20;
    size_t localArraySize = (sizeToSum/comSize);
    int* localArray = malloc(localArraySize*sizeof(int));
    for(int i = 0; i < localArraySize; i++)
        localArray[i] = rank*localArraySize + i;

    uint64_t result;
    MPI_P2P_reduce(localArray, &result, localArraySize, MPI_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
        printf("%lu\n", result);

    free(localArray);
    MPI_Finalize();
    return 0;
}