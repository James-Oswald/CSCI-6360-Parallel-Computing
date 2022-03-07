
#ifndef REDUCE_CUDA_H
#define REDUCE_CUDA_H

#include<stddef.h>

typedef struct ReductionInfo{
    size_t localArraySize;      //size of the array to reduce
    double* deviceLocalArray;   //pointer to GPU copy of the local array
    double* deviceTempResult;  //pointer to GPU intermediate result buffer
    double* deviceResult;      //pointer to GPU reduction results
} ReductionInfo;

#ifdef __cplusplus
extern "C" {
#endif

ReductionInfo cudaInit(int rank, const double* localArray, size_t localArraySize);
double cudaReduce(ReductionInfo* ri);

#ifdef __cplusplus
}
#endif

#endif