/*********************************************************************/
//
// 02/01/2022: Revised Version for 32M bit adder with 32 bit blocks
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 8388608 // hex digits 
#define block_size 32
#define verbose 1

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

int* dsumi;
int* dbin1;
int* dbin2;

int sumi[bits] = {0}; //Change this back to be of size bits
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


__global__ void compute_gp(const int* b1, const int* b2, int* g, int* p){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	g[i] = b1[i] & b2[i];
	p[i] = b1[i] | b2[i];
}

/*__global__ void compute_g(int* gGroup, int* pGroup){
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
}*/

__global__ void compute_chunk_gp(const int* subchunkg, const int* subchunkp, int* chunkg, int* chunkp){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	int start = i*block_size;
	const int* gGroup = subchunkg+start; 				//pointer to the group of 32 g's this thread is reducing
	const int* pGroup = subchunkp+start; 	    //pointer to the group of 32 p's this thread is reducing
	
	//computing the generate
	int sum = 0;
	for(int j = 0; j < block_size; j++){
		int mult = gGroup[j];
		for(int k = block_size-1; k > i; k--)
			mult &= pGroup[k];
		sum |= mult;
	}
	chunkg[i] = sum;

	//computing the prop
	int mult = subchunkp[0];
	for(int j = 1; j < block_size; j++)
		mult &= pGroup[j];
	chunkp[i] = mult;
}

__global__ void compute_chunk_carry(const int* subchunkg, const int* subchunkp, const int* chunkc, int* subchunkc){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	int carry = i == 0 ? 0 : chunkc[i-1];
	for(int j = i*block_size; j < (i+1)*block_size; j++)
		carry = (subchunkc[j] = subchunkg[j] | (subchunkp[j] & carry));
}

__global__ void compute_sum(const int* b1, const int* b2, const int* c, int* sum){
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	int carry = i == 0 ? 0 : c[i-1];
	sum[i] = b1[i] ^ b2[i] ^ carry;
}

void cringePrint(int* thing, int size, const char* frmt = "%s\n"){
	char* thingStr = int_to_string(thing, size);
	char* rev = revbinary_to_hex(thingStr, size);
	printf(frmt, rev);
}

void cla(){
	const int nthreads = 32;
	compute_gp<<<bits,nthreads>>>(dbin1, dbin2, gi, pi);
	compute_chunk_gp<<<ngroups,nthreads>>>(gi, pi, ggj, gpj);
	compute_chunk_gp<<<nsections,nthreads>>>(ggj, gpj, sgk, spk);
	compute_chunk_gp<<<nsupersections,nthreads>>>(sgk, spk, ssgl, sspl);
	compute_chunk_gp<<<nsupersupersections,nthreads>>>(ssgl, sspl, sssgm, ssspm);
	compute_chunk_carry<<<1,1>>>(sssgm, ssspm, nullptr, ssscm);
	compute_chunk_carry<<<nsupersupersections,nthreads>>>(ssgl, sspl, ssscm, sscl);
	compute_chunk_carry<<<nsupersections,nthreads>>>(sgk, spk, sscl, sck);
	compute_chunk_carry<<<nsections,nthreads>>>(ggj, gpj, sck, gcj);
	compute_chunk_carry<<<ngroups,nthreads>>>(gi, pi, gcj, ci);
	compute_sum<<<bits,nthreads>>>(dbin1, dbin2, ci, dsumi);
	cudaDeviceSynchronize();
	int hsumi[bits];
	cudaMemcpy(hsumi, dsumi, bits*sizeof(int), cudaMemcpyDeviceToHost);
	//cringePrint(hsumi, bits, "%.20s\n");
}

void ripple_carry_adder(const int size){
	int clast=0, cnext=0;
	for(int i = 0; i < size; i++){
		cnext = (bin1[i] & bin2[i]) | ((bin1[i] | bin2[i]) & clast);
		sumrca[i] = bin1[i] ^ bin2[i] ^ clast;
		clast = cnext;
	}
}

void check_cla_rca(const int size){
	for(int i = 0; i < size; i++){
		if( sumrca[i] != sumi[i]){
			printf("Check: Found sumrca[%d] = %d, not equal to sumi[%d] = %d - stopping check here!\n",
				i, sumrca[i], i, sumi[i]);
			//printf("bin1[%d] = %d, bin2[%d]=%d, gi[%d]=%d, pi[%d]=%d, ci[%d]=%d, ci[%d]=%d\n",
			//	i, bin1[i], i, bin2[i], i, gi[i], i, pi[i], i, ci[i], i-1, ci[i-1]);
			return;
		}
	}
	printf("Check Complete: CLA and RCA are equal\n");
}



/*

void babyCLA(){	

	cudaMallocManaged((void**)&ssgl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sspl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sscl, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&sssgm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssspm, nsupersupersections*sizeof(int));
	cudaMallocManaged((void**)&ssscm, nsupersupersections*sizeof(int));
	const int nthreads = 1;
	
	compute_gp<<<nsupersections,nthreads>>>(dbin1, dbin2, ssgl, sspl);
	cudaDeviceSynchronize();
	int hssgl[nsupersections];
	int hsspl[nsupersections];
	cudaMemcpy(hssgl, ssgl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hsspl, sspl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hssgl, nsupersections, "G1: %s\n");
	cringePrint(hsspl, nsupersections, "P1: %s\n");
	
	compute_chunk_gp<<<nsupersupersections, nthreads, 2*nsupersections>>>(nsupersupersections, ssgl, sspl, sssgm, ssspm);
	cudaDeviceSynchronize();
	int hsssgm[nsupersupersections];
	int hssspm[nsupersupersections];
	cudaMemcpy(hsssgm, sssgm, nsupersupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hssspm, ssspm, nsupersupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hsssgm, nsupersupersections, "G2: %s\n");
	cringePrint(hssspm, nsupersupersections, "P2: %s\n");
	/*cudaMemcpy(hssgl, ssgl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hsspl, sspl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hssgl, nsupersections, "G1: %s\n");
	cringePrint(hsspl, nsupersections, "P1: %s\n");
	
	compute_chunk_carry<<<1,nthreads>>>(1, sssgm, ssspm, nullptr, ssscm);
	cudaDeviceSynchronize();
	int hssscm[nsupersupersections];
	cudaMemcpy(hssscm, ssscm, nsupersupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hssscm, nsupersupersections, "C1: %s\n");
	
	/*cudaMemcpy(hssgl, ssgl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hsspl, sspl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hssgl, nsupersections, "G1: %s\n");
	cringePrint(hsspl, nsupersections, "P1: %s\n");

	compute_chunk_carry<<<nsupersupersections,nthreads>>>(nsupersupersections, ssgl, sspl, ssscm, sscl);
	cudaDeviceSynchronize();
	int hsscl[nsupersections];
	cudaMemcpy(hsscl, sscl, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hsscl, nsupersections, "C2: %s\n");

	int hbin1[nsupersections];
	int hbin2[nsupersections];
	cudaMemcpy(hbin1, dbin1, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hbin2, dbin2, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hbin1, nsupersections, "B1: %s\n");
	cringePrint(hbin2, nsupersections, "B2: %s\n");

	compute_sum<<<nsupersections,nthreads>>>(dbin1, dbin2, sscl, dsumi);
	cudaDeviceSynchronize();
	int hsumi[nsupersections];
	cudaMemcpy(hsumi, dsumi, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	cringePrint(hsumi, nsupersections, "S0: %s\n");
}

int main(){
	int size = 32*32;

	//char* hex1 = generate_random_hex(size);
	//char* hex2 = generate_random_hex(size);
	read_input(size);
	char* hexa = prepend_non_sig_zero(hex1);
	char* hexb = prepend_non_sig_zero(hex2);
	bin1 = gen_formated_binary_from_hex(hexa);
	bin2 = gen_formated_binary_from_hex(hexb);
	cudaMallocManaged((void**)&dbin1, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&dbin2, nsupersections*sizeof(int));
	cudaMallocManaged((void**)&dsumi, nsupersections*sizeof(int));
	cudaMemcpy(dbin1, bin1, nsupersections*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbin2, bin2, nsupersections*sizeof(int), cudaMemcpyHostToDevice);
	babyCLA();
	cudaMemcpy(sumi, dsumi, nsupersections*sizeof(int), cudaMemcpyDeviceToHost);
	ripple_carry_adder(nsupersections);
	check_cla_rca(nsupersections);

	cringePrint(sumi, nsupersections);
	cringePrint(sumrca, nsupersections);
}*/


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
	cudaMallocManaged((void**)&dsumi, bits*sizeof(int));
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
	cudaMemcpy(dbin1, bin1, bits*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbin2, bin2, bits*sizeof(int), cudaMemcpyHostToDevice);
	start_time = clock_now();
	cla();
	end_time = clock_now();
	cudaMemcpy(sumi, dsumi, bits*sizeof(int), cudaMemcpyDeviceToHost);
	printf("CLA Completed in %llu cycles\n", (end_time - start_time));

	start_time = clock_now();
	ripple_carry_adder(bits);
	end_time = clock_now();

	printf("RCA Completed in %llu cycles\n", (end_time - start_time));

	check_cla_rca(bits);

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
