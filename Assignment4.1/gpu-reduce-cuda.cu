
#include<cstdio>


//Shared memory helper for reduce 7
template <class T>
struct SharedMemory {
    __device__ inline operator T *(){
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const{
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

//double specialized shared memory helper for reduce 7
template <>
struct SharedMemory<double> {
    __device__ inline operator double *() {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

//helper for reduce 7
template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum){
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        mySum += __shfl_down_sync(mask, mySum, offset);
    return mySum;
}

//The main cuda reduction kernel, sums up each block
template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata, unsigned int n){
    T *sdata = SharedMemory<T>();
    // perform first level of reduction, reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * gridDim.x;
    unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;
    T mySum = 0;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    if (nIsPow2){
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        gridSize = gridSize << 1;
        while (i < n){
            mySum += g_idata[i];
            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if ((i + blockSize) < n)
                mySum += g_idata[i + blockSize];
            i += gridSize;
        }
    }else{
        unsigned int i = blockIdx.x * blockSize + threadIdx.x;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
    }

    // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH == SM 8.0
    mySum = warpReduceSum<T>(mask, mySum);
    // each thread puts its local sum into shared memory
    if ((tid % warpSize) == 0)
        sdata[tid / warpSize] = mySum;

    __syncthreads();

    const unsigned int shmem_extent = (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
    const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
    if (tid < shmem_extent) {
        mySum = sdata[tid];
        // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH == SM 8.0
        mySum = warpReduceSum<T>(ballot_result, mySum);
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

//globals for this cuda translation unit, no need to extern them, we use cudaInit to set them 
// and cudaReduce makes use of them.
size_t localArrayLength;
double* d_localArray;
double* d_outputArray;

//init cuda, set device, allocate memory, fill local array.
extern "C" void cudaInit(int rank, size_t localArrayLength_){
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
    localArrayLength = localArrayLength_;
    cudaMallocManaged(&d_localArray, localArrayLength*sizeof(double));
    //We know this will be smaller than the localArrayLength but dont know blocksize in cuda init.
    cudaMallocManaged(&d_outputArray, localArrayLength*sizeof(double));

    for(size_t i = 0; i < localArrayLength; i++)
        d_localArray[i] =  rank*localArrayLength + i;
}

//The reduction function takes the number of threads and the number of blocks, uses the globals set in cudaInit
//It calls the kernal then preforms a CPU sum over the block sums as was described in class, returning the total
// local sum over all blocks
extern "C" double cudaReduce(int threads, int blocks){
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = ((threads / 32) + 1) * sizeof(double);
    //threads are effectively locked at 512 here since we need them to be known at compile time
    reduce7<double, 512, false><<<dimGrid, dimBlock, smemSize>>>(d_localArray, d_outputArray, localArrayLength);
    cudaDeviceSynchronize();
    double localSum = 0;
    for(int i = 0; i < blocks; i++) //preform the final reduction on the CPU
        localSum += d_outputArray[i];
    return localSum;
}
