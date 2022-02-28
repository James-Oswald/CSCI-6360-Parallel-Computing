
#include<stdio.h>
#include<stdint.h>


int main(){
    //1073741824
    uint64_t sum = 0;
    for(uint64_t i = 0; i < 2<<20; i++)
        sum += i;
    printf("%lu\n", sum);
    //printf(sum == 576460751766552576 ? "eq\n" : "!eq\n"); 
}

