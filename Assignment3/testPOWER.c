
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

int main(int argc, char** argv){
	assert(argc == 3);
	int pow = atoi(argv[1]);
	int comSize = atoi(argv[2]);
	size_t localArraySize = (2<<pow)/comSize;
	long* localArray = (long*)malloc(localArraySize*sizeof(long));
	printf("Malloced Pointer for 2^%d longs: %lu\n", pow, localArray);
	if(!localArray)
		perror("Error:");
	else
		free(localArray);
}
