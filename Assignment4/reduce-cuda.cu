

extern "C" 
void cudaInit(int rank, const double* localArray, size_t localArraySize, double** dLocalArray, double** dResult){
    cudaError_t cE;
    if((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess){
        printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }
    if((cE = cudaSetDevice(rank % cudaDeviceCount)) != cudaSuccess){
        printf("Unable to have rank %d set to cuda device %d, error is %d \n", 
                rank, (rank%cudaDeviceCount), cE);
        exit(-1);
    }
    cudaMallocManaged((void**)&deviceLocalArray, localArraySize*sizeof(double));
    cudaMemcpy(dbin2, localArray, bits*sizeof(int), cudaMemcpyHostToDevice);
}

extern "C"
void cudaReduce(const double* dLocalArray, double* result, uint64_t* startTime, uint64_t* endTime){
    
}