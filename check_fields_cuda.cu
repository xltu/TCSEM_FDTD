/*
 !=====================================================================
 !
 !               		TCSEM_FDTD	version  2.0
 !               ---------------------------------------
 ! This is the CUDA version of the TCSEM_FDTD program
 !
 ! TCSEM_FDTD is a time domain modeling code for marine controlled source
 ! electromagnetic method. The code models the impulse response generated 
 ! by a electrical bipole souce towed in the seawater. All six components 
 ! (i.e., Ex, Ey, Ez, Bx, By, and Bz) of the EM field could be obtained 
 ! simultaneously from one forward modeling.
 !
 ! The FDTD modeling uses dufort frankel scheme, and staggered grid
 ! The source was added in a similar way as in seismic modeling
 ! The waveform of a delt function is approximated by a Gaussian function
 ! The output of the programs is the impulse response
 !
 ! TODO In the current version, multiple transmitter positions are Looped 
 ! on one GPU device squentially. This should be changed to parallel
 ! threads on multiple GPU devices in feature version
 !
 ! XXX Check README to prepare the input files required by this program
 !
 ! Created Dec.28.2020 by Xiaolei Tu
 ! Send bug reports, comments or suggestions to tuxl2009@hotmail.com
 !=====================================================================
 */

/* The current header file is used to deal with the GPU device and
	CUDA version related issues.
   It is modified from 'check_fields_cuda.h' from the program 'specfem3d 
   version 3.0' at https://github.com/geodynamics/specfem3d
*/   

#include "Cuda_device.h"


/* ----------------------------------------------------------------------------------------------- */

// GPU device memory functions

/* ----------------------------------------------------------------------------------------------- */

void get_free_memory(double* free_db, double* used_db, double* total_db) {

  TRACE("get_free_memory");

  // gets memory usage in byte
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
  if (cudaSuccess != cuda_status){
    printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
    exit(EXIT_FAILURE);
  }

  *free_db = (double)free_byte ;
  *total_db = (double)total_byte ;
  *used_db = *total_db - *free_db ;
  return;
}

/* ----------------------------------------------------------------------------------------------- */

// Saves GPU memory usage to file
void output_free_memory(int myrank, char* info_str) {

  TRACE("output_free_memory");

  FILE* fp;
  char filename[BUFSIZ];
  double free_db,used_db,total_db;
  int do_output_info;

  // by default, only main process outputs device infos to avoid file cluttering
  do_output_info = 0;
  if (myrank == 0){
    do_output_info = 1;
    sprintf(filename,OUTPUT_FILES"/gpu_device_mem_usage.txt");
  }
  // debugging
  if (DEBUG){
    do_output_info = 1;
    sprintf(filename,OUTPUT_FILES"/gpu_device_mem_usage_proc_%06d.txt",myrank);
  }

  // outputs to file
  if (do_output_info){

    // gets memory usage
    get_free_memory(&free_db,&used_db,&total_db);

    // file output
    fp = fopen(filename,"a+");
    if (fp != NULL){
      fprintf(fp,"%d: @%s GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n", myrank, info_str,
              used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
      fclose(fp);
    }
  }
}

/* ----------------------------------------------------------------------------------------------- */
void output_free_device_memory(int* myrank) {

  TRACE("output_free_device_memory");

  char info_str[15]; // add extra character for null termination
  int len;

  // safety check to avoid string buffer overflow
  if (*myrank > 99999999) { exit_on_error("Error: rank too large in output_free_device_memory() routine"); }

  len = snprintf (info_str, 15, "rank %8d:", *myrank);
  if (len >= 15){ printf("warning: string length truncated (from %d) in output_free_device_memory() routine\n", len); }

  //debug
  //printf("debug: info ***%s***\n",info_str);

  // writes to output file
  output_free_memory(*myrank, info_str);
}


/* ----------------------------------------------------------------------------------------------- */
void get_free_device_memory(realw* free, realw* used, realw* total) {
  TRACE("get_free_device_memory");

  double free_db,used_db,total_db;

  get_free_memory(&free_db,&used_db,&total_db);

  // converts to MB
  *free = (realw) free_db/1024.0/1024.0;
  *used = (realw) used_db/1024.0/1024.0;
  *total = (realw) total_db/1024.0/1024.0;
  return;
}



/* ----------------------------------------------------------------------------------------------- */

// Auxiliary functions

/* ----------------------------------------------------------------------------------------------- */

/*
__global__ void memset_to_realw_kernel(realw* array, int size, realw value){

  unsigned int tid = threadIdx.x;
  unsigned int bx = blockIdx.y*gridDim.x+blockIdx.x;
  unsigned int i = tid + bx*blockDim.x;

  if (i < size){
    array[i] = *value;
  }
}
*/

/* ----------------------------------------------------------------------------------------------- */

realw get_device_array_maximum_value(realw* array, int size){

// get maximum of array on GPU by copying over to CPU and handle it there

  realw max = 0.0f;

  // checks if anything to do
  if (size > 0){
    realw* h_array;

    // explicitly wait for cuda kernels to finish
    // (cudaMemcpy implicitly synchronizes all other cuda operations)
    synchronize_cuda();

    h_array = (realw*)calloc(size,sizeof(realw));
    print_CUDA_error_if_any(cudaMemcpy(h_array,array,sizeof(realw)*size,cudaMemcpyDeviceToHost),33001);

    // finds maximum value in array
    max = h_array[0];
    for( int i=1; i < size; i++){
      if (abs(h_array[i]) > max) max = abs(h_array[i]);
    }
    free(h_array);
  }
  return max;
}




