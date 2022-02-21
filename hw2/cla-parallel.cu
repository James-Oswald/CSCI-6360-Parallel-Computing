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

//pointers to device memory
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

int* dsumi;
int* dbin1;
int* dbin2;

//host side
int sumi[bits] = {0};
int sumrca[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input(int size){
	char* in1 = (char *)calloc(size+1, sizeof(char));
	char* in2 = (char *)calloc(size+1, sizeof(char));
	if( 1 != scanf("%s", in1)){
		printf("Failed to read input 1\n");
		exit(-1);
	}
	if( 1 != scanf("%s", in2)){
		printf("Failed to read input 2\n");
		exit(-1);
	}
	hex1 = grab_slice_char(in1,0,size+1);
	hex2 = grab_slice_char(in2,0,size+1);
	free(in1);
	free(in2);
}

//This function computes the gs and ps for all bits
__global__ void compute_gp(const int* b1, const int* b2, int* g, int* p){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=bits) 
		return; //Throw out any aditional threads 
	g[i] = b1[i] & b2[i];
	p[i] = b1[i] | b2[i];
}

//This is a generic kernal to compute the generates and propagates at any level 
//nchunks is the number of bits in each section, subchunkg and subchunkp are the gs and ps
//of the previous levels. chunkg and chunkp are the gs and ps computed from from block_size subchunk gs and ps. 
template<int nchunks>
__global__ void compute_chunk_gp(const int* subchunkg, const int* subchunkp, int* chunkg, int* chunkp){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=nchunks) 
		return; //Throw out any aditional threads 
	int start = i*block_size;
	const int* gGroup = subchunkg+start; 	//pointer to the group of 32 g's this thread is reducing
	const int* pGroup = subchunkp+start; 	//pointer to the group of 32 p's this thread is reducing
	
	//computing the generate
	int sum = 0;
	for(int j = 0; j < block_size; j++){
		int mult = gGroup[j];
		for(int k = block_size-1; k > j; k--)
			mult &= pGroup[k];
		sum |= mult;
	}
	chunkg[i] = sum;

	//computing the prop
	int mult = pGroup[0];
	for(int j = 1; j < block_size; j++)
		mult &= pGroup[j];
	chunkp[i] = mult;
}


//This is a generic kernal to compute carries at any level.
//It takes the list of generates and list of propagates of the current level and the carries of one level up
//and computes the carries of the current chunk (subchunkc). nchunks is a generic paramater that 
template<int nchunks>
__global__ void compute_chunk_carry(const int* subchunkg, const int* subchunkp, const int* chunkc, int* subchunkc){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=nchunks) return;
	int carry = i == 0 ? 0 : chunkc[i-1];
	for(int j = i*block_size; j < (i+1)*block_size; j++){
		subchunkc[j] = subchunkg[j] | (subchunkp[j] & carry);
		carry = subchunkc[j];
	}
}

__global__ void compute_sum(const int* b1, const int* b2, const int* c, int* sum){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=bits) return;
	int carry = i == 0 ? 0 : c[i-1];
	sum[i] = b1[i] ^ b2[i] ^ carry;
}

void cla(){
	//Allocate all of the GPU memory we're going to use
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

	const int nthreads = 32; //The number of threads used, vary this for preformance report

	//Compute top level gs and ps
	compute_gp<<<bits,nthreads>>>(dbin1, dbin2, gi, pi);

	//Compute gps for each layer
	compute_chunk_gp<ngroups><<<ngroups,nthreads>>>(gi, pi, ggj, gpj);
	compute_chunk_gp<nsections><<<nsections,nthreads>>>(ggj, gpj, sgk, spk);
	compute_chunk_gp<nsupersections><<<nsupersections,nthreads>>>(sgk, spk, ssgl, sspl);
	compute_chunk_gp<nsupersupersections><<<nsupersupersections,nthreads>>>(ssgl, sspl, sssgm, ssspm);
	
	//Compute carries for each layer	
	compute_chunk_carry<1><<<1,nthreads>>>(sssgm, ssspm, nullptr, ssscm);
	compute_chunk_carry<nsupersupersections><<<nsupersupersections,nthreads>>>(ssgl, sspl, ssscm, sscl);
	compute_chunk_carry<nsupersections><<<nsupersections,nthreads>>>(sgk, spk, sscl, sck);
	compute_chunk_carry<nsections><<<nsections,nthreads>>>(ggj, gpj, sck, gcj);
	compute_chunk_carry<ngroups><<<ngroups,nthreads>>>(gi, pi, gcj, ci);

	//finally compute the sum
	compute_sum<<<bits,nthreads>>>(dbin1, dbin2, ci, dsumi);
	
	//ensure that all the values have been written to dsumi via a device sync
	cudaDeviceSynchronize();
}

//This is the built in RCA computation
void ripple_carry_adder(){
	int clast=0, cnext=0;
	for(int i = 0; i < bits; i++){
		cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
		sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
		clast = cnext;
	}
}

//This is the built in RCA comparison
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
		read_input(input_size);
	}else{
		srand(deterministic_seed);
		hex1 = generate_random_hex(input_size);
		hex2 = generate_random_hex(input_size);
	}
	
	hexa = prepend_non_sig_zero(hex1);
	hexb = prepend_non_sig_zero(hex2);
	hexa[digits] = '\0'; //double checking
	hexb[digits] = '\0';
	
	bin1 = gen_formated_binary_from_hex(hexa);
	bin2 = gen_formated_binary_from_hex(hexb);

	//Allocate device memory for the inputs and outputs
	cudaMallocManaged((void**)&dbin1, bits*sizeof(int));
	cudaMallocManaged((void**)&dbin2, bits*sizeof(int));
	cudaMallocManaged((void**)&dsumi, bits*sizeof(int));

	//Copy inputs over to the device
	cudaMemcpy(dbin1, bin1, bits*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbin2, bin2, bits*sizeof(int), cudaMemcpyHostToDevice);
	
	start_time = clock_now();
	cla();
	end_time = clock_now();

	//Copy outputs back client side for checking
	cudaMemcpy(sumi, dsumi, bits*sizeof(int), cudaMemcpyDeviceToHost);
	printf("CLA Completed in %llu cycles\n", (end_time - start_time));

	start_time = clock_now();
	ripple_carry_adder();
	end_time = clock_now();

	printf("RCA Completed in %llu cycles\n", (end_time - start_time));

	check_cla_rca();

	if( verbose==1 ){
		int2str_result = int_to_string(sumi,bits);
		hexSum = revbinary_to_hex(int2str_result,bits);
		//printf("%.20s\n",hexSum);
		//int2str_result = int_to_string(sumrca,bits);
		//hexSum = revbinary_to_hex(int2str_result,bits);
		//printf("%.20s\n",hexSum);
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
		//print_chararrayln(hexa);
		printf("b   ");
		//print_chararrayln(hexb);
	}
	if ( verbose==1 ){
		printf("Hex Return\n");
		printf("sum =  ");
	}
	
	// free memory from prepend call
	free(hexa);
	free(hexb);

	if( verbose==1 ){
		//printf(".20%s\n",hexSum);
		//printf(".20%s\n",hexSum);
	}
	
	free(hexSum);
	
	return 1;
}
