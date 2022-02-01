
//James Oswald

/*********************************************************************/
//
// Created by Chander Iyer and Neil McGlohon.
//
/*********************************************************************/

#include "main.h"

//Touch these defines
#define input_size 1024 // hex digits 
#define block_size 8
#define verbose 0

//Do not touch these defines
#define digits (input_size+1)
#define bits (digits*4)
#define ngroups bits/block_size
#define nsections ngroups/block_size
#define nsupersections nsections/block_size

//Global definitions of the various arrays used in steps for easy access
int gi[bits] = {0};
int pi[bits] = {0};
int ci[bits] = {0};

int ggj[ngroups] = {0};
int gpj[ngroups] = {0};
int gcj[ngroups] = {0};

int sgk[nsections] = {0};
int spk[nsections] = {0};
int sck[nsections] = {0};

int ssgl[nsupersections] = {0} ;
int sspl[nsupersections] = {0} ;
int sscl[nsupersections] = {0} ;

int sumi[bits] = {0};

//Integer array of inputs in binary form
int* bin1=NULL;
int* bin2=NULL;

//Character array of inputs in hex form
char* hex1=NULL;
char* hex2=NULL;

void read_input(){
    char* in1 = calloc(input_size+1, sizeof(char));
    char* in2 = calloc(input_size+1, sizeof(char));
    scanf("%s", in1);
    scanf("%s", in2);
    hex1 = grab_slice_char(in1,0,input_size);
    hex2 = grab_slice_char(in2,0,input_size);
    free(in1);
    free(in2);
}

//Computes the generate and propagate of for each individual bit
void compute_gp(){
    for(int i = 0; i < bits; i++){ //This loop can be run in parallel
        gi[i] = bin1[i] & bin2[i];
        pi[i] = bin1[i] | bin2[i];
    }
}

//Computes the generate and propagate values for each group of 8 bits
void compute_group_gp(){
    for(int j = 0; j < ngroups; j++){
        int jstart = j*block_size;

        //get groups of bit generate and propagate values
        int* ggj_group = grab_slice(gi,jstart,block_size);
        int* gpj_group = grab_slice(pi,jstart,block_size);
        
        //Compute group generate
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = ggj_group[i]; //grabs the g_i term for the multiplication
            for(int ii = block_size-1; ii > i; ii--){
                mult &= gpj_group[ii]; //grabs the p_i terms and multiplies it with the previously multiplied stuff (or the g_i term if first round)
            }
            sum |= mult; //sum up each of these things with an or
        }
        ggj[j] = sum;

        //Compute group propagate
        int mult = gpj_group[0];
        for(int i = 1; i < block_size; i++){
            mult &= gpj_group[i];
        }
        gpj[j] = mult;

        //free from grab_slice allocation
        free(ggj_group);
        free(gpj_group);
    }
}

//Computes the generate and propagate values for each section of 8 groups
//Algo is the same as compute_group_gp, see comments there
void compute_section_gp(){
    for(int k = 0; k < nsections; k++){
        int kstart = k*block_size;
        int* sgk_group = grab_slice(ggj, kstart, block_size); 
        int* spk_group = grab_slice(gpj, kstart, block_size);
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = sgk_group[i]; 
            for(int ii = block_size-1; ii > i; ii--){
                mult &= spk_group[ii]; 
            }
            sum |= mult; 
        }
        sgk[k] = sum;

        int mult = spk_group[0];
        for(int i = 1; i < block_size; i++){
            mult &= spk_group[i];
        }
        spk[k] = mult;
        free(sgk_group);
        free(spk_group);
    }
}

//Computes the generate and propagate values for each super section of 8 sections
//Algo is the same as compute_group_gp, see comments there
void compute_super_section_gp(){
    for(int l = 0; l < nsupersections; l++){
        int lstart = l*block_size;
        int* ssgl_group = grab_slice(sgk, lstart, block_size); 
        int* sspl_group = grab_slice(spk, lstart, block_size);
        int sum = 0;
        for(int i = 0; i < block_size; i++){
            int mult = ssgl_group[i]; 
            for(int ii = block_size-1; ii > i; ii--){
                mult &= sspl_group[ii]; 
            }
            sum |= mult;
        }
        ssgl[l] = sum;

        int mult = sspl_group[0];
        for(int i = 1; i < block_size; i++){
            mult &= sspl_group[i];
        }
        sspl[l] = mult;
        free(ssgl_group);
        free(sspl_group);
    }
}

//Computes the carry bit for each super section
void compute_super_section_carry(){
    for(int l = 0; l < nsupersections; l++) //This can not be parallelized due to dependence on sscl[l-1]
        sscl[l] = ssgl[l] | (sspl[l] & (l==0 ? 0 : sscl[l-1]));
}

//Computes the carry bit for each section within each super section. 
void compute_section_carry(){
    for(int l = 0; l < nsupersections; l++) //This is the loop that gets parallelized
        for(int k = l*block_size; k < (l+1)*block_size; k++) 
            sck[k] = sgk[k] | (spk[k] & (k%block_size==0 ? (l==0 ? 0: sscl[l-1]) : sck[k-1]));
}

//Computes the carry bit for each group within each section. 
void compute_group_carry(){
    for(int k = 0; k < nsections; k++)  //This is the loop that gets parallelized
        for(int j = k*block_size; j < (k+1)*block_size; j++) 
            gcj[j] = ggj[j] | (gpj[j] & (j%block_size==0 ? (k==0 ? 0 : sck[k-1]) : gcj[j-1]));
}

//Computes the carry bit for each bit within each group. 
void compute_carry(){
    for(int j = 0; j < ngroups; j++)  //This is the loop that gets parallelized
        for(int i = j*block_size; i < (j+1)*block_size; i++)
            ci[i] = gi[i] | (pi[i] & (i%block_size==0 ? (j==0 ? 0 : gcj[j-1]) : ci[i-1]));
}

//Compute the final sum using the xor addition formula wrt carries
void compute_sum(){
    for(int i = 0; i < bits; i++) //This loop could be parallelized
        sumi[i] = bin1[i] ^ bin2[i] ^ (i==0 ? 0 : ci[i-1]);
}

void cla(){
    compute_gp();
    compute_group_gp();
    compute_section_gp();
    compute_super_section_gp();
    compute_super_section_carry();
    compute_section_carry();
    compute_group_carry();
    compute_carry();
    compute_sum();
}

int main(int argc, char *argv[])
{
    int randomGenerateFlag = 0;
    char* hexa=NULL;
    char* hexb=NULL;
    char* hexSum=NULL;
    char* int2str_result=NULL;  

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
        hex1 = generate_random_hex(input_size);
        hex2 = generate_random_hex(input_size);
    }

    hexa = prepend_non_sig_zero(hex1);
    hexb = prepend_non_sig_zero(hex2);
    hexa[digits] = '\0'; //double checking
    hexb[digits] = '\0';

    bin1 = gen_formated_binary_from_hex(hexa);
    bin2 = gen_formated_binary_from_hex(hexb);

    cla();

    int2str_result = int_to_string(sumi,bits);
    hexSum = revbinary_to_hex( int2str_result,bits);
    // hexSum = revbinary_to_hex(int_to_string(sumi,bits),bits);
    // free inputs fields allocated in read_input or gen random calls
    free(int2str_result);
    free(hex1);
    free(hex2);

    // free bin conversion of hex inputs
    free(bin1);
    free(bin2);
    
    if(verbose==1 || randomGenerateFlag==1)
    {
        printf("Hex Input\n");
        printf("a   ");
        print_chararrayln(hexa);
        printf("b   ");
        print_chararrayln(hexb);
    }
    if (verbose==1 || randomGenerateFlag==1)
    {
        printf("Hex Return\n");
        printf("sum =  ");
    }

    // free memory from prepend call
    free(hexa);
    free(hexb);
    printf("%s\n",hexSum);
    free(hexSum);
    return 1;
}
