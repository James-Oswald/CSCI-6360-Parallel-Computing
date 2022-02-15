/*********************************************************************/
//
// 02/01/2022: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size
#define nsupersupersections nsupersections/block_size

//Global definitions of the various arrays used in steps for easy access
int* gi;
int* pi; 
int* ci;

int* ggj;
int* gpj;
int* gcj;

int* sgk;
int* spk;
int* sck;

int* ssgl;
int* sspl;
int* sscl;

int* ssspm;
int* sssgm;
int* ssscm;

int* dbin1;
int* dbin2;
int* dsumi;

int* sumi;

int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input(){
	char* in1 = (char *)calloc(input_size+1, sizeof(char));
	char* in2 = (char *)calloc(input_size+1, sizeof(char));
	if( 1 != scanf("%s", in1)){
		printf("Failed to read input 1\n");
		exit(-1);
	}
	if( 1 != scanf("%s", in2)){
		printf("Failed to read input 2\n");
		exit(-1);
	}
	hex1 = grab_slice_char(in1,0,input_size+1);
	hex2 = grab_slice_char(in2,0,input_size+1);
	free(in1);
	free(in2);
}


__global__ void compute_gp(int* b1, int* b2, int* g, int* p){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    g[i] = b1[i] & b2[i];
    p[i] = b1[i] | b2[i];
}

__device__ compute_g(int* gGroup, int* pGroup){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	for(int ii = block_size-1; ii > i; ii--)
		gGroup[i] &= pGroup[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
	__syncthreads();
	if(i < 16){
		gGroup[i] |= gGroup[i + 16];
		gGroup[i] |= gGroup[i + 8];
		gGroup[i] |= gGroup[i + 4];
		gGroup[i] |= gGroup[i + 1];
	}
}

__device__ compute_p(int* pGroup){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < 16){
		pGroup[i] |= pGroup[i + 16];
		pGroup[i] |= pGroup[i + 8];
		pGroup[i] |= pGroup[i + 4];
		pGroup[i] |= pGroup[i + 1];
	}
}

__global__ void compute_chunk_gp(int* subchunkg, int* subchunkp, int* chunkg, int* chunkp){
	__shared__ int gshared[blockDim.x * block_size];
	__shared__ int pshared[blockDim.x * block_size];
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	gshared[index] = subchunkg[index];
	pshared[index] = subchunkp[index];
	int start = index*block_size;
	int* gGroup = gshared+start; //pointer to the group of 32 g's this thread is reducing
	int* pGroup = pshared+start; //pointer to the group of 32 p's this thread is reducing
	__syncthreads();
	compute_g<<<1,block_size>>>(gGroup, pGroup);
	chunkg[index] = gGroup[0];
	compute_p<<<1,block_size>>>(pGroup);
	chunkp[index] = pGroup[0];
}

__global__ void compute_sss_carry(int* sssg, int* sssp, int* sssc){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	for(int )
	sssc[i] = sssg[i] | (sssp[i] & (i==0 ? 0 : sscl[l-1]));
}

__global__ void compute_chunk_carry(int* chunkg, int* chunkp, int* chunkc, int* subchunkc){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	for(int j = i*block_size; j < (i+1)*block_size; j++){
		subchunkc[j] = 
	}
}

void cla(){
	cudaMallocManaged((void**)&dsumi, bits);
	cudaMallocManaged((void**)&dbin1, bits);
	cudaMallocManaged((void**)&dbin2, bits);
	cudaMallocManaged((void**)&gi, bits);
	cudaMallocManaged((void**)&pi, bits);
	cudaMallocManaged((void**)&ci, bits);
	cudaMallocManaged((void**)&ggj, ngroups);
	cudaMallocManaged((void**)&gpj, ngroups);
	cudaMallocManaged((void**)&gcj, ngroups);
	cudaMallocManaged((void**)&sgk, nsections);
	cudaMallocManaged((void**)&spk, nsections);
	cudaMallocManaged((void**)&sck, nsections);
	cudaMallocManaged((void**)&ssgl, nsupersections);
	cudaMallocManaged((void**)&sspl, nsupersections);
	cudaMallocManaged((void**)&sscl, nsupersections);
	cudaMallocManaged((void**)&sssgm, nsupersupersections);
	cudaMallocManaged((void**)&ssspm, nsupersupersections);
	cudaMallocManaged((void**)&ssscm, nsupersupersections);
	cudaMemcpy((void*)dbin1, bin1, bits, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)dbin2, bin2, bits, cudaMemcpyHostToDevice);

	const int nthreads = 32;
    compute_gp<<<bits,nthreads>>>(dbin1, dbin2, gi, pi);
    compute_chunk_gp<<<ngroups,nthreads>>>(gi, pi, ggj, gpj);
	compute_chunk_gp<<<nsections,nthreads>>>(ggj, gpj, sgk, spk);
	compute_chunk_gp<<<nsupersections,nthreads>>>(sgk, spk, ssgl, sspl);
	compute_chunk_gp<<<nsupersupersections,nthreads>>>(ssgl, sspl, sssgm, ssspm);

    compute_section_gp();
    compute_super_section_gp();
    compute_super_super_section_gp();
    compute_super_super_section_carry();
    compute_super_section_carry();
    compute_section_carry();
    compute_group_carry();
    compute_carry();
    compute_sum();
}

void ripple_carry_adder()
{
  int clast=0, cnext=0;

  for(int i = 0; i < bits; i++)
    {
      cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
      sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
      clast = cnext;
    }
}

void check_cla_rca()
{
  for(int i = 0; i < bits; i++)
    {
      if( sumrca[i] != sumi[i] )
  {
    printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
     i, sumrca[i], i, sumi[i]);
    printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
     i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
    return;
  }
    }
  printf("Check Complete: CLA and RCA are equal\n");
}

int main(int argc, char *argv[])
{
  int randomGenerateFlag = 1;
  int deterministic_seed = (1<<30) - 1;
  char* hexa=NULL;
  char* hexb=NULL;
  char* hexSum=NULL;
  char* int2str_result=NULL;
  unsigned long long start_time=clock_now(); // dummy clock reads to init
  unsigned long long end_time=clock_now();   // dummy clock reads to init

  if( nsupersupersections != block_size )
    {
      printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
       nsupersupersections, block_size );
      return(-1);
    }
  
  if (argc == 2) {
    if (strcmp(argv[1], "-r") == 0)
      randomGenerateFlag = 1;
  }
  
  if (randomGenerateFlag == 0)
    {
      read_input();
    }
  else
    {
      srand( deterministic_seed );
      hex1 = generate_random_hex(input_size);
      hex2 = generate_random_hex(input_size);
    }
  
  hexa = prepend_non_sig_zero(hex1);
  hexb = prepend_non_sig_zero(hex2);
  hexa[digits] = '\0'; //double checking
  hexb[digits] = '\0';
  
  bin1 = gen_formated_binary_from_hex(hexa);
  bin2 = gen_formated_binary_from_hex(hexb);

  start_time = clock_now();
  cla();
  end_time = clock_now();

  printf("CLA Completed in %llu cycles\n", (end_time - start_time));

  start_time = clock_now();
  ripple_carry_adder();
  end_time = clock_now();

  printf("RCA Completed in %llu cycles\n", (end_time - start_time));

  check_cla_rca();

  if( verbose==1 )
    {
      int2str_result = int_to_string(sumi,bits);
      hexSum = revbinary_to_hex( int2str_result,bits);
    }

  // free inputs fields allocated in read_input or gen random calls
  free(int2str_result);
  free(hex1);
  free(hex2);
  
  // free bin conversion of hex inputs
  free(bin1);
  free(bin2);
  
  if( verbose==1 )
    {
      printf("Hex Input\n");
      printf("a   ");
      print_chararrayln(hexa);
      printf("b   ");
      print_chararrayln(hexb);
    }
  
  if ( verbose==1 )
    {
      printf("Hex Return\n");
      printf("sum =  ");
    }
  
  // free memory from prepend call
  free(hexa);
  free(hexb);

  if( verbose==1 )
    printf("%s\n",hexSum);
  
  free(hexSum);
  
  return 1;
}
