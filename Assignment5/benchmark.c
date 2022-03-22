
#include<mpi.h>

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>

#ifdef __x86_64__
#include<time.h>
const int clockFrequency = CLOCKS_PER_SEC;
unsigned long long clockRead(){
    return clock();
}
#else
const int clockFrequency = 512000000;
unsigned long long clockRead(){
    unsigned int tbl, tbu0, tbu1;
    do{
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    }while(tbu0 != tbu1);
    return (((unsigned long long)tbu0) << 32) | tbl;
}
#endif

const int numBlocks = 32;
const int K = 1<<10, M = 1<<20;    //block size bases (in binary NOT decimal?)
//#define lenBlockSizes 8
//const int blockSizes[lenBlockSizes] = {128*K, 256*K, 512*K, 1*M, 2*M, 4*M, 8*M, 16*M};
#define lenBlockSizes 3
const int blockSizes[lenBlockSizes] = {128*K, 256*K, 1*M};

int rank, worldSize;
uint8_t* data;   //The buffer of all 1s that is written and subsequently read

void benchmark(const char* name, const char* filePath){
    
    if(rank == 0)
        printf("Benchmarking for %s:\n", name);
    
    for(int i = 0; i < lenBlockSizes; i++){ //For each blocksize

        //set up our data and open a file
        size_t blockSize = blockSizes[i];
        data = (uint8_t*)malloc(blockSize);
        memset(data, 0xff, blockSize);      //Fill block with ones
        char newFilePath[260];
        sprintf(newFilePath, "%s%lf.bin", filePath, blockSize/(double)M);
        MPI_File fileHandle; 
        MPI_File_open(MPI_COMM_WORLD, newFilePath, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fileHandle);

        if(rank == 0)
            printf("\tBlocksize %lf MB, Filesize: %lf MB:\n", blockSize/(double)M, (blockSize*worldSize*32)/(double)M);
        //Don't start timing until all ranks have init the data and read the file
        MPI_Barrier(MPI_COMM_WORLD); 
        unsigned long long start = clockRead();

        for(int j = 0; j < 32; j++){ //each rank writes 32 blocks
            //Which block we're writing to, in line with the scheme in the pdf: rank 0 block 0, etc..
            size_t blockIndex = rank+worldSize*j; 
            MPI_File_write_at(fileHandle, blockIndex*blockSize, data, blockSize, MPI_UINT8_T, MPI_STATUS_IGNORE);
        }

        //Finish the timing only after all ranks are done with writing
        MPI_Barrier(MPI_COMM_WORLD); 
        unsigned long long end = clockRead();
        if(rank == 0){
            double writeTime = (end-start)/(double)clockFrequency;
            printf("\t\tWrite:\n\t\t\t%lf Seconds\n\t\t\t%lf MB/S\n", writeTime, ((blockSize*worldSize*32)/(double)M)/writeTime);
        }

        //Don't start timing for read until the print is done
        MPI_Barrier(MPI_COMM_WORLD); 
        start = clockRead();

        for(int j = 0; j < 32; j++){ //each rank reads 32 blocks
            //Which block we're reading from, in line with the scheme in the pdf: rank 0 block 0, etc..
            size_t blockIndex = rank+worldSize*j; 
            MPI_File_read_at(fileHandle, blockIndex*blockSize, data, blockSize, MPI_UINT8_T, MPI_STATUS_IGNORE);
        }

        //Finish the timing only after all ranks are done with reading
        MPI_Barrier(MPI_COMM_WORLD); 
        end = clockRead();
        if(rank == 0){
            double readTime = (end-start)/(double)clockFrequency;
            printf("\t\tRead:\n\t\t\t%lf Seconds\n\t\t\t%lf MB/S\n", readTime, ((blockSize*worldSize*32)/(double)M)/readTime);
        }

        //close and delete the file before testing the next block size
        MPI_File_close(&fileHandle);
        MPI_Barrier(MPI_COMM_WORLD); 
        if(rank == 0){
            MPI_File_delete(newFilePath, MPI_INFO_NULL);
        }
        free(data);
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    benchmark("Disk", "/mnt/d/program/CSCI-6360-Parallel-Computing/Assignment5/test");
    benchmark("SSD", "/mnt/c/Users/James/test");
    //benchmark("scratch", "//");
    //benchmark("NVMe", "/");
    MPI_Finalize();
    return 0;
}

