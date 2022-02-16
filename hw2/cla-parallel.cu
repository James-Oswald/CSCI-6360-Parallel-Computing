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

//int* sumi;
int sumi[bits] = {0};
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

__global__ void compute_g(int* gGroup, int* pGroup){
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

__global__ void compute_p(int* pGroup){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < 16){
		pGroup[i] |= pGroup[i + 16];
		pGroup[i] |= pGroup[i + 8];
		pGroup[i] |= pGroup[i + 4];
		pGroup[i] |= pGroup[i + 1];
	}
}

__global__ void compute_chunk_gp(size_t chunkSize, int* subchunkg, int* subchunkp, int* chunkg, int* chunkp){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= chunkSize)
		return;
	int start = i*block_size;
	int* gGroup = subchunkg+start; //pointer to the group of 32 g's this thread is reducing
	int* pGroup = subchunkp+start; //pointer to the group of 32 p's this thread is reducing
	compute_g<<<1,block_size>>>(gGroup, pGroup);
	chunkg[i] = gGroup[0];
	__syncthreads(); //maybe not needed?
	compute_p<<<1,block_size>>>(pGroup);
	chunkp[i] = pGroup[0];
}

__global__ void compute_chunk_carry(size_t chunkSize, int* subchunkg, int* subchunkp, int* chunkc, int* subchunkc){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= chunkSize)
		return;
	int carry = i == 0 ? 0 : chunkc[i-1];
	for(int j = i*block_size+1; j < (i+1)*block_size; j++)
		carry = subchunkc[j] = subchunkg[j] | (subchunkp[j] & carry);
}

__global__ void compute_sum(int* b1, int* b2, int* c, int* sum){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	int carry = i == 0 ? 0 : c[i-1];
	sum[i] = b1[i] ^ b2[i] ^ carry;
}


void cla(){
	//cudaMallocManaged((void**)&sumi, bits*sizeof(int));
	cudaMallocManaged((void**)&dsumi, bits);
	cudaMallocManaged((void**)&gi, bits*sizeof(int));
	cudaMallocManaged((void**)&pi, bits*sizeof(int));
	cudaMallocManaged((void**)&ci, bits*sizeof(int));
	cudaMallocManaged((void**)&ggj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&gpj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&gcj, ngroups*sizeof(int));
	cudaMallocManaged((void**)&sgk, nsections*sizeof(int));
	cudaMallocManaged((void**)&spk, nsections*sizeof(int));
	cudaMallocManaged((void**)&sck, nsections*sizeof(int));
	cudaMallocManaged((void**)&ssgl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sspl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sscl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sssgm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssspm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssscm, nsupersupersections*sizeof(int));

	const int nthreads = 1;
	compute_gp<<<bits,nthreads>>>(dbin1, dbin2, gi, pi);
	//compute_gp<<<bits,nthreads>>>(bin1, bin2, gi, pi);

	compute_chunk_gp<<<ngroups,nthreads>>>(ngroups, gi, pi, ggj, gpj);
	compute_chunk_gp<<<nsections,nthreads>>>(nsections, ggj, gpj, sgk, spk);
	compute_chunk_gp<<<nsupersections,nthreads>>>(nsupersections, sgk, spk, ssgl, sspl);
	compute_chunk_gp<<<nsupersupersections,nthreads>>>(nsupersupersections, ssgl, sspl, sssgm, ssspm);
	//printf("%s/n", int_to_string(sssgm, nsupersupersections));
	//printf("%s/n", int_to_string(ssspm, nsupersupersections));
	compute_chunk_carry<<<1,nthreads>>>(1, sssgm, ssspm, nullptr, ssscm);
	compute_chunk_carry<<<nsupersupersections,nthreads>>>(nsupersupersections, ssgl, sspl, ssscm, sscl);
	compute_chunk_carry<<<nsupersections,nthreads>>>(nsupersections, sgk, spk, sscl, sck);
	compute_chunk_carry<<<nsections,nthreads>>>(nsections, ggj, gpj, sck, gcj);
	compute_chunk_carry<<<ngroups,nthreads>>>(ngroups, gi, pi, gcj, ci);
	
	compute_sum<<<bits,nthreads>>>(dbin1, dbin2, ci, dsumi);
	//compute_sum<<<bits,nthreads>>>(bin1, bin2, ci, sumi);
	cudaDeviceSynchronize();
}

void ripple_carry_adder(){
	int clast=0, cnext=0;
	for(int i = 0; i < bits; i++){
		cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
		sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
		clast = cnext;
	}
}

void check_cla_rca(){
	for(int i = 0; i < bits; i++){
		if( sumrca[i] != sumi[i]){
			printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
				i, sumrca[i], i, sumi[i]);
			printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
				i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
			return;
		}
	}
	printf("Check Complete: CLA and RCA are equal\n");
}


int main(int argc, char *argv[]){
	int randomGenerateFlag = 1;
	int deterministic_seed = (1<<30) - 1;
	char* hexa=NULL;
	char* hexb=NULL;
	char* hexSum=NULL;
	char* int2str_result=NULL;
	unsigned long long start_time=clock_now(); // dummy clock reads to init
	unsigned long long end_time=clock_now();   // dummy clock reads to init

	if(nsupersupersections != block_size){
			printf("Misconfigured CLA - nsupersupersections (%d) not equal to block_size (%d) \n",
				nsupersupersections, block_size );
			return(-1);
	}
	
	if (argc == 2) {
		if (strcmp(argv[1], "-r") == 0)
			randomGenerateFlag = 1;
	}
	
	if (randomGenerateFlag == 0){
		read_input();
	}else{
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


	cudaMallocManaged((void**)&dbin1, bits*sizeof(int));
	cudaMallocManaged((void**)&dbin2, bits*sizeof(int));
	cudaMemcpy(dbin1, bin1, bits*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbin2, bin2, bits*sizeof(int), cudaMemcpyHostToDevice);
	start_time = clock_now();
	cla();
	end_time = clock_now();
	cudaMemcpy(sumi, dsumi, bits*sizeof(int), cudaMemcpyDeviceToHost);

	printf("CLA Completed in %llu cycles\n", (end_time - start_time));

	start_time = clock_now();
	ripple_carry_adder();
	end_time = clock_now();

	printf("RCA Completed in %llu cycles\n", (end_time - start_time));

	check_cla_rca();

	if( verbose==1 ){
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
	
	if( verbose==1 ){
		printf("Hex Input\n");
		printf("a   ");
		print_chararrayln(hexa);
		printf("b   ");
		print_chararrayln(hexb);
	}
	if ( verbose==1 ){
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
