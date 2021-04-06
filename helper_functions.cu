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
   It is modified from 'helper_functions.cu' from the program 'specfem3d 
   version 3.0' at https://github.com/geodynamics/specfem3d
*/  
#include <cufft.h>
#include <cusparse.h>
#include "Cuda_device.h"
#ifdef WITH_MPI
#include <mpi.h>
#endif
/* ----------------------------------------------------------------------------------------------- */

// Helper functions

/* ----------------------------------------------------------------------------------------------- */

// copies integer array from CPU host to GPU device
void copy_todevice_int(void** d_array_addr_ptr,int* h_array,int size){
  TRACE("  copy_todevice_int");

  // allocates memory on GPU
  //
  // note: cudaMalloc uses a double-pointer, such that it can return an error code in case it fails
  //          we thus pass the address to the pointer above (as void double-pointer) to have it
  //          pointing to the correct pointer of the array here
  print_CUDA_error_if_any(cudaMalloc((void**)d_array_addr_ptr,size*sizeof(int)),12001);

  // copies values onto GPU
  //
  // note: cudaMemcpy uses the pointer to the array, we thus re-cast the value of
  //          the double-pointer above to have the correct pointer to the array
  print_CUDA_error_if_any(cudaMemcpy((int*) *d_array_addr_ptr,h_array,size*sizeof(int),cudaMemcpyHostToDevice),12002);
}

/* ----------------------------------------------------------------------------------------------- */

// copies integer array from CPU host to GPU device
void copy_todevice_realw(void** d_array_addr_ptr,realw* h_array,int size){
  TRACE("  copy_todevice_realw");

  // allocates memory on GPU
  print_CUDA_error_if_any(cudaMalloc((void**)d_array_addr_ptr,size*sizeof(realw)),22001);

  // copies values onto GPU
  print_CUDA_error_if_any(cudaMemcpy((realw*) *d_array_addr_ptr,h_array,size*sizeof(realw),cudaMemcpyHostToDevice),22002);
}


/* ----------------------------------------------------------------------------------------------- */

// CUDA synchronization

/* ----------------------------------------------------------------------------------------------- */

void synchronize_cuda() {
#if CUDA_VERSION < 4000 || (defined (__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 4))
    cudaThreadSynchronize();
#else
    cudaDeviceSynchronize();
#endif
}

/* ----------------------------------------------------------------------------------------------- */

void print_CUDA_error_if_any(cudaError_t err, int num) {
  if (cudaSuccess != err)
  {
    printf("\nCUDA error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",cudaGetErrorString(err),num);
    fflush(stdout);

    // outputs error file
    FILE* fp;
    int myrank;
    char filename[BUFSIZ];
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
    myrank = 0;
#endif
    sprintf(filename,OUTPUT_FILES"/error_message_%06d.txt",myrank);
    fp = fopen(filename,"a+");
    if (fp != NULL){
      fprintf(fp,"\nCUDA error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",cudaGetErrorString(err),num);
      fclose(fp);
    }

    // stops program
#ifdef WITH_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(EXIT_FAILURE);
  }
}

// for simplicity
#define HANDLE_ERROR( err ) (print_CUDA_error_if_any( err, NULL ))

/*----------------------------------------------------------------------------------------------------*/
//		cufft errors
/*---------------------------------------------------------------------------------------------------*/
// https://docs.nvidia.com/cuda/cufft/index.html#cufftresult
static const char *_cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "The plan parameter is not a valid handle";

        case CUFFT_ALLOC_FAILED:
            return "The allocation of GPU or CPU memory for the plan failed";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "One or more invalid parameters were passed to the API";

        case CUFFT_INTERNAL_ERROR:
            return "An internal driver error was detected";

        case CUFFT_EXEC_FAILED:
            return "cuFFT failed to execute the transform on the GPU";

        case CUFFT_SETUP_FAILED:
            return "The cuFFT library failed to initialize";

        case CUFFT_INVALID_SIZE:
            return "One or more of the parameters is not a supported size";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
            
        case CUFFT_INCOMPLETE_PARAMETER_LIST:    
        		return "Missing parameters in call";
       
       	case CUFFT_INVALID_DEVICE:
       			return "An invalid GPU index was specified in a descriptor or Execution of a plan was on different GPU than plan creation";  
       			
       	case CUFFT_PARSE_ERROR:
       			return "Internal plan database error";		
       	
       	case CUFFT_NO_WORKSPACE:
       			return "No workspace has been provided prior to plan execution";		
       	
       	case CUFFT_NOT_IMPLEMENTED:
       			return "Function does not implement functionality for parameters given";		 
       			   
       	case CUFFT_LICENSE_ERROR:
       			return "Used in previous versions";	
       			
       	case CUFFT_NOT_SUPPORTED:
       			return "Operation is not supported for parameters given";			
    }

    return "<unknown>";
}

void print_cuFFT_error_if_any(cufftResult err, int num) {
  if (CUFFT_SUCCESS != err)
  {
    printf("\ncuFFT error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",_cufftGetErrorEnum(err),num);
    fflush(stdout);

    // outputs error file
    FILE* fp;
    int myrank;
    char filename[BUFSIZ];
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
    myrank = 0;
#endif
    sprintf(filename,OUTPUT_FILES"/error_message_%06d.txt",myrank);
    fp = fopen(filename,"a+");
    if (fp != NULL){
      fprintf(fp,"\ncuFFT error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",_cufftGetErrorEnum(err),num);
      fclose(fp);
    }

    // stops program
#ifdef WITH_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(EXIT_FAILURE);
  }
}

/*----------------------------------------------------------------------------------------------------*/
//		cusparse errors
/*---------------------------------------------------------------------------------------------------*/
// https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-types-reference

void print_cuSparse_error_if_any(cusparseStatus_t err, int num) {
  if (CUSPARSE_STATUS_SUCCESS != err)
  {
    printf("\ncusparse error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",cusparseGetErrorString(err),num);
    fflush(stdout);

    // outputs error file
    FILE* fp;
    int myrank;
    char filename[BUFSIZ];
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
    myrank = 0;
#endif
    sprintf(filename,OUTPUT_FILES"/error_message_%06d.txt",myrank);
    fp = fopen(filename,"a+");
    if (fp != NULL){
      fprintf(fp,"\ncusparse error !!!!! <%s> !!!!! \nat CUDA call error code: # %d\n",cusparseGetErrorString(err),num);
      fclose(fp);
    }

    // stops program
#ifdef WITH_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(EXIT_FAILURE);
  }
}


/* ----------------------------------------------------------------------------------------------- */
// Timing helper functions
/* ----------------------------------------------------------------------------------------------- */

void start_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop){
  // creates & starts event
  cudaEventCreate(start);
  cudaEventCreate(stop);
  cudaEventRecord( *start, 0 );
}

/* ----------------------------------------------------------------------------------------------- */

void stop_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop, const char* info_str){
  float time;
  // stops events
  cudaEventRecord( *stop, 0 );
  cudaEventSynchronize( *stop );
  cudaEventElapsedTime( &time, *start, *stop );
  cudaEventDestroy( *start );
  cudaEventDestroy( *stop );
  // user output
  printf("%s: Execution Time = %f ms\n",info_str,time);
}

/* ----------------------------------------------------------------------------------------------- */

void stop_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop, const char* info_str, realw* t){
  float time;
  // stops events
  cudaEventRecord( *stop, 0);
  cudaEventSynchronize( *stop );
  cudaEventElapsedTime( &time, *start, *stop );
  cudaEventDestroy( *start );
  cudaEventDestroy( *stop );
  // user output
  printf("%s: Execution Time = %f ms\n",info_str,time);

  // returns time
  *t = time;
}


/* ----------------------------------------------------------------------------------------------- */

void exit_on_cuda_error(const char* kernel_name) {
  //check to catch errors from previous operations

  // synchronizes GPU
  synchronize_cuda();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr,"GPU Error: after %s: %s\n", kernel_name, cudaGetErrorString(err));

    //debugging
    //pause_for_debugger(0);

    // outputs error file
    FILE* fp;
    int myrank;
    char filename[BUFSIZ];
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
    myrank = 0;
#endif
    sprintf(filename,OUTPUT_FILES"/error_message_%06d.txt",myrank);
    fp = fopen(filename,"a+");
    if (fp != NULL){
      fprintf(fp,"GPU Error: after %s: %s\n", kernel_name, cudaGetErrorString(err));
      fclose(fp);
    }

    // stops program
    //free(kernel_name);
#ifdef WITH_MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    exit(EXIT_FAILURE);
  }
}

/* ----------------------------------------------------------------------------------------------- */

void exit_on_error(const char* info) {
  printf("\nERROR: %s\n",info);
  fflush(stdout);

  // outputs error file
  FILE* fp;
  int myrank;
  char filename[BUFSIZ];
#ifdef WITH_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
  myrank = 0;
#endif
  sprintf(filename,OUTPUT_FILES"/error_message_%06d.txt",myrank);
  fp = fopen(filename,"a+");
  if (fp != NULL){
    fprintf(fp,"ERROR: %s\n",info);
    fclose(fp);
  }

  // stops program
#ifdef WITH_MPI
  MPI_Abort(MPI_COMM_WORLD,1);
#endif
  //free(info);
  exit(EXIT_FAILURE);
}


/*----------------------------------------------------------------------------------------------- */

// additional helper functions

/*----------------------------------------------------------------------------------------------- */

double get_time_val() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec*1e-6;
}

/* ----------------------------------------------------------------------------------------------- */

void pause_for_debugger(int pause) {
  if (pause) {
    int myrank;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#else
    myrank = 0;
#endif
    printf("I'm rank %d\n",myrank);
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s:%d ready for attach\n", getpid(), hostname,myrank);

    FILE *file = fopen("./attach_gdb.txt","w+");
    if (file != NULL){
      fprintf(file,"PID %d on %s:%d ready for attach\n", getpid(), hostname,myrank);
      fclose(file);
    }

    fflush(stdout);
    while (0 == i)
      sleep(5);
  }
}


// MPI synchronization

/* ----------------------------------------------------------------------------------------------- */

void synchronize_mpi () {
#ifdef WITH_MPI
  MPI_Barrier (MPI_COMM_WORLD);
#endif
}


/* ----------------------------------------------------------------------------------------------- */

// for debugging purposes, unused so far...

/* ----------------------------------------------------------------------------------------------- */

//extern EXTERN_LANG
//void FC_FUNC_(fortranflush,FORTRANFLUSH)(int* rank){
//TRACE("fortranflush");
//
//  fflush(stdout);
//  fflush(stderr);
//  printf("Flushing proc %d!\n",*rank);
//}
//
//extern EXTERN_LANG
//void FC_FUNC_(fortranprint,FORTRANPRINT)(int* id) {
//TRACE("fortranprint");
//
//  int procid;
//#ifdef WITH_MPI
//  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
//#else
//  procid = 0;
//#endif
//  printf("%d: sends msg_id %d\n",procid,*id);
//}
//
//extern EXTERN_LANG
//void FC_FUNC_(fortranprintf,FORTRANPRINTF)(realw* val) {
//TRACE("fortranprintf");
//
//  int procid;
//#ifdef WITH_MPI
//  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
//#else
//  procid = 0;
//#endif
//  printf("%d: sends val %e\n",procid,*val);
//}
//
//extern EXTERN_LANG
//void FC_FUNC_(fortranprintd,FORTRANPRINTD)(double* val) {
//TRACE("fortranprintd");
//
//  int procid;
//#ifdef WITH_MPI
//  MPI_Comm_rank(MPI_COMM_WORLD,&procid);
//#else
//  procid = 0;
//#endif
//  printf("%d: sends val %e\n",procid,*val);
//}
