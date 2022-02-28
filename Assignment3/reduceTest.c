#define _GNU_SOURCE
#include<mpi.h>
#include<stdio.h>
#include<sched.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    printf("cpu=%d\n", sched_getcpu());

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int data[2] = {1, 2};
    int sum[2];
    MPI_Reduce(&data, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
        printf("%d\n", sum);

    MPI_Finalize();
    return 0;
}