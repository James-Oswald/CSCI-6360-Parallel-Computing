
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>

#define localArraySize ((2<<30)/2)

int main(int argc, char** argv){
	long localArray[localArraySize];
	for(int i = 0; i < localArraySize; i++)
		localArray[i] = 2*localArraySize + i;
	printf("Done Lmao \n");
	return 0;
}
