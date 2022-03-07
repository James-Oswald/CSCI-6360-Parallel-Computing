

#include<cstdio>
#include"reduce-cuda.h"

#define numThreads 256
#define numBlocks 64

__device__ __forceinline__ double warpReduceSum(unsigned int mask, double mySum){
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        mySum += __shfl_down_sync(mask, mySum, offset);
    return mySum;
}

#define nIsPow2 true
__global__ void reduce7(const double*__restrict__ g_idata, double*__restrict__ g_odata, unsigned int size){
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int blockSize = blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;

    double mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    if(nIsPow2){
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        gridSize = gridSize << 1;
        while (i < size){
            mySum += g_idata[i];
            // ensure we don't read out of bounds -- this is optimized away for
            // powerOf2 sized arrays
            if ((i + blockSize) < size)
                mySum += g_idata[i + blockSize];
            i += gridSize;
        }
    }else{
        unsigned int i = blockIdx.x * blockSize + threadIdx.x;
        while (i < size){
            mySum += g_idata[i];
            i += gridSize;
        }
    }

    // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum(mask, mySum);

    // each thread puts its local sum into shared memory
    if ((tid % warpSize) == 0)
        sdata[tid / warpSize] = mySum;

    __syncthreads();

    const unsigned int shmem_extent = (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent){
        mySum = sdata[tid];
        // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH == SM 8.0
        mySum = warpReduceSum(ballot_result, mySum);
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

extern "C" 
ReductionInfo cudaInit(int rank, const double* localArray, const size_t localArraySize){
    cudaError_t cE;
    int cudaDeviceCount;
    cE = cudaGetDeviceCount(&cudaDeviceCount);
    if(cE != cudaSuccess){
        printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }
    cE = cudaSetDevice(rank % cudaDeviceCount);
    if(cE != cudaSuccess){
        printf("Unable to have rank %d set to cuda device %d, error is %d \n", rank, (rank%cudaDeviceCount), cE);
        exit(-1);
    }
    ReductionInfo reductionInfo;
    cudaMallocManaged((void**)&reductionInfo.deviceLocalArray, localArraySize*sizeof(double));
    cudaMemcpy(reductionInfo.deviceLocalArray, localArray, localArraySize*sizeof(double), cudaMemcpyHostToDevice);
    reductionInfo.localArraySize = localArraySize;
    cudaMallocManaged((void**)&reductionInfo.deviceResult, (localArraySize/numBlocks)*sizeof(double));
    cudaMallocManaged((void**)&reductionInfo.deviceTempResult, localArraySize*sizeof(double));
    return reductionInfo;
}

extern "C" 
double cudaReduce(ReductionInfo* ri){
    int threads = numThreads;
    int blocks = numBlocks;
    int smemSize = ((threads / 32) + 1) * sizeof(double); 
    reduce7<<<blocks, threads, smemSize>>>(ri->deviceLocalArray, ri->deviceResult, ri->localArraySize);
    int s = blocks; //s is the size of ri->deviceTempResult
    while(s > 1){
        blocks = (s + (threads * 2 - 1)) / (threads * 2);
        cudaMemcpy(ri->deviceTempResult, ri->deviceResult, s*sizeof(double), cudaMemcpyDeviceToDevice);
        reduce7<<<blocks, threads, numThreads/32>>>(ri->deviceTempResult, ri->deviceResult, s);
        s = (s + (threads * 2 - 1)) / (threads * 2);
    }
    double result;
    cudaMemcpy(&result, ri->deviceResult, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}
