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
 ! XXX Check README to prepare the input files required by this program
 !
 ! Created Dec.28.2020 by Xiaolei Tu
 ! Send bug reports, comments or suggestions to tuxl2009@hotmail.com
 !=====================================================================
 */
 
/* The current header file is used to deal with costants in GPU device memory
   It is modified from 'prepare_constants_cuda.h' from the program 'specfem3d 
   version 3.0' at https://github.com/geodynamics/specfem3d
*/ 

//TODO replace the exit(1) in this section with a more general
//	function to include the MPI-CUDA hybrid situation, but for
//	now it is OK to use exit

#ifndef PREPARE_CONSTANTS_CUDA_H
#define PREPARE_CONSTANTS_CUDA_H

#include"FTDT.h"

// macros for version output
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var " = "  VALUE(var)

#pragma message ("Compiling with: " VAR_NAME_VALUE(CUDA_VERSION) "\n")
#if defined(__CUDA_ARCH__)
#pragma message ("Compiling with: " VAR_NAME_VALUE(__CUDA_ARCH__) "\n")
#endif

// CUDA version >= 5.0 needed for new symbol addressing and texture binding
#if CUDA_VERSION < 5000
  #ifndef USE_OLDER_CUDA4_GPU
    #define USE_OLDER_CUDA4_GPU
  #endif
#else
  #undef USE_OLDER_CUDA4_GPU
#endif

#ifdef USE_OLDER_CUDA4_GPU
#pragma message ("\nCompiling with: USE_OLDER_CUDA4_GPU enabled\n")
#endif

/* ----------------------------------------------------------------------------------------------- */

// CONSTANT arrays setup

/* ----------------------------------------------------------------------------------------------- */

/* note:
 constant arrays when used in compute_forces_acoustic_cuda.cu routines stay zero,
 constant declaration and cudaMemcpyToSymbol would have to be in the same file...

 extern keyword doesn't work for __constant__ declarations.

 also:
 cudaMemcpyToSymbol("deviceCaseParams", caseParams, sizeof(CaseParams));
 ..
 and compile with -arch=sm_20

 see also: http://stackoverflow.com/questions/4008031/how-to-use-cuda-constant-memory-in-a-programmer-pleasant-way
 doesn't seem to work.

 for now, we store pointers with cudaGetSymbolAddress() function calls.

 */

// cuda constant variables
//
//__constant__ realw miu_=4*PI*1.e-7;
#pragma once

__constant__ ModelPara mp_;
//__constant__ realw lambda;
//__constant__ realw	dt;
__constant__ TxPos Tx_;
	
ModelPara* setConst_MP(ModelPara* Host_mp)
{
  cudaError_t err = cudaMemcpyToSymbol(mp_, Host_mp, sizeof(ModelPara));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error in setConst_mp: %s\n", cudaGetErrorString(err));
    exit(1);
  }
 	
 	ModelPara *d_mp;
#ifdef USE_OLDER_CUDA4_GPU
  err = cudaGetSymbolAddress((void**)&(d_mp),"mp_");
#else
  err = cudaGetSymbolAddress((void**)&(d_mp),mp_);
#endif
  if (err != cudaSuccess) {
    fprintf(stderr, "Error with d_mp: %s\n", cudaGetErrorString(err));
    exit(1);
  } 	 
	
	return d_mp;
}

/*
realw* setConst_dt(realw Host_dt)
{

  cudaError_t err = cudaMemcpyToSymbol(dt, &Host_dt, sizeof(realw));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error in setConst_dt: %s\n", cudaGetErrorString(err));
    exit(1);
  }
	
	realw *d_dt;
#ifdef USE_OLDER_CUDA4_GPU
  err = cudaGetSymbolAddress((void**)&(d_dt),"dt");
#else
  err = cudaGetSymbolAddress((void**)&(d_dt),dt);
#endif
  if (err != cudaSuccess) {
    fprintf(stderr, "Error with d_dt: %s\n", cudaGetErrorString(err));
    exit(1);
  } 	 	
  
  return d_dt;
}

realw* setConst_lambda(realw Host_lambda)
{

  cudaError_t err = cudaMemcpyToSymbol(lambda, &Host_lambda, sizeof(realw));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error in setConst_lambda: %s\n", cudaGetErrorString(err));
    exit(1);
  }
	
	realw *d_lambda;
#ifdef USE_OLDER_CUDA4_GPU
  err = cudaGetSymbolAddress((void**)&(d_lambda),"lambda");
#else
  err = cudaGetSymbolAddress((void**)&(d_lambda),lambda);
#endif
  if (err != cudaSuccess) {
    fprintf(stderr, "Error with d_lambda: %s\n", cudaGetErrorString(err));
    exit(1);
  } 	 
  
  return d_lambda;	
}

*/

TxPos* setConst_Tx(TxPos* Host_Tx)
{
  cudaError_t err = cudaMemcpyToSymbol(Tx_, Host_Tx, sizeof(TxPos));
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error in setConst_Tx: %s\n", cudaGetErrorString(err));
    exit(1);
  }
 	
 	TxPos *d_Tx;
#ifdef USE_OLDER_CUDA4_GPU
  err = cudaGetSymbolAddress((void**)&(d_Tx),"Tx_");
#else
  err = cudaGetSymbolAddress((void**)&(d_Tx),Tx_);
#endif
  if (err != cudaSuccess) {
    fprintf(stderr, "Error with d_Tx: %s\n", cudaGetErrorString(err));
    exit(1);
  } 	 
	
	return d_Tx;
}


#endif //PREPARE_CONSTANTS_CUDA_H
