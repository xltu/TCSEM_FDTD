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
   It is modified from 'mesh_constants_cuda.h' from the program 'specfem3d 
   version 3.0' at https://github.com/geodynamics/specfem3d
*/   	


/* trivia

- for most working arrays we use now "realw" instead of "float" type declarations to make it easier to switch
  between a real or double precision simulation
  (matching CUSTOM_REAL == 4 or 8 in fortran routines).

- instead of boolean "logical" declared in fortran routines, in C (or Cuda-C) we have to use "int" variables.
  ifort / gfortran caveat:
    to check whether it is true or false, do not check for == 1 to test for true values since ifort just uses
    non-zero values for true (e.g. can be -1 for true). however, false will be always == 0.
  thus, rather use: if (var ) {...}  for testing if true instead of if (var == 1){...} (alternative: one could use if (var != 0){...}

*/

#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include "FTDT.h"


// performance timers
#define CUDA_TIMING 0
#define CUDA_TIMING_UPDATE 0

// error checking after cuda function calls
// (note: this synchronizes many calls, thus e.g. no asynchronuous memcpy possible)
#define ENABLE_VERY_SLOW_ERROR_CHECKING 1
#if ENABLE_VERY_SLOW_ERROR_CHECKING == 1
#define GPU_ERROR_CHECKING(x) exit_on_cuda_error(x);
#else
#define GPU_ERROR_CHECKING(x)
#endif

/* ----------------------------------------------------------------------------------------------- */
// cuda constant arrays
/* ----------------------------------------------------------------------------------------------- */

// dimensions
#define NDIM 3

// Output paths, see setup/constants.h
#define OUTPUT_FILES "./OUTPUT_FILES/"

/* ----------------------------------------------------------------------------------------------- */

// (optional) pre-processing directive used in kernels: if defined check that it is also set in setup/constants.h:
// leads up to ~ 5% performance increase
//#define USE_MESH_COLORING_GPU

/* ----------------------------------------------------------------------------------------------- */

// Texture memory usage:
// requires CUDA version >= 4.0, see check below
// Use textures for d_displ and d_accel -- 10% performance boost
//#define USE_TEXTURES_FIELDS

// Using texture memory for the hprime-style constants is slower on
// Fermi generation hardware, but *may* be faster on Kepler
// generation.
// Use textures for hprime_xx
//#define USE_TEXTURES_CONSTANTS

// CUDA version >= 4.0 needed for cudaTextureType1D and cudaDeviceSynchronize()
#if CUDA_VERSION < 4000
#undef USE_TEXTURES_FIELDS
#undef USE_TEXTURES_CONSTANTS
#endif

#ifdef USE_TEXTURES_FIELDS
#pragma message ("\nCompiling with: USE_TEXTURES_FIELDS enabled\n")
#endif
#ifdef USE_TEXTURES_CONSTANTS
#pragma message ("\nCompiling with: USE_TEXTURES_CONSTANTS enabled\n")
#endif

// (optional) unrolling loops
// leads up to ~1% performance increase
//#define MANUALLY_UNROLLED_LOOPS
//
//											performance statistics: main kernels: 
//
//	UpdateEx();
//	Used 48 registers, 480 bytes cmem[0], 48 bytes cmem[2]
//
//	UpdateEy():
//	Used 52 registers, 3096 bytes smem, 480 bytes cmem[0], 48 bytes cmem[2]
//
//  UpdateEz():
//	Used 52 registers, 3096 bytes smem, 472 bytes cmem[0], 48 bytes cmem[2]
//
//	UpdateBx():
//	Used 40 registers, 2176 bytes smem, 432 bytes cmem[0], 48 bytes cmem[2]
//
//	UpdateBy():
//	Used 40 registers, 2176 bytes smem, 432 bytes cmem[0], 48 bytes cmem[2]
//
//	UpdateBz():
//	Used 41 registers, 4352 bytes smem, 424 bytes cmem[0], 48 bytes cmem[2]
//
// CUDA compiler specifications
// (optional) use launch_bounds specification to increase compiler optimization
// (depending on GPU type, register spilling might slow down the performance)
// 
#ifdef GPU_DEVICE_Kepler 
// specifics see: https://docs.nvidia.com/cuda/kepler-tuning-guide/index.html
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
// maximum shared memory per thread block 48KB 
// Total SMEM 49152 per sm
// Maximum Threads per Block: 1024
// Maximum resident blocks per SM: 16
// Maximum warps per sm: 64 
//	To achieve Max capacity of Kepler, 
//		smem should be limited to 6144 B per block for 16*16 thread block
//															7072 B							16*8 t
//		Total Register 65536
//		register per thread should be limited to 32 per thread			 												
//	Register pressue is the main factor
// (uncomment if desired)
//#define USE_LAUNCH_BOUNDS
#define LAUNCH_MIN_BLOCKS 10
//#pragma message ("\nCompiling with: USE_LAUNCH_BOUNDS enabled for K20\n")
#endif

// add more card specific values
#ifdef GPU_DEVICE_Maxwell
// specifics see: https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html
// register file size 64k 32-bit registers per SM
// shared memory 64KB for GM107 and 96KB for GM204
#undef USE_LAUNCH_BOUNDS
#endif

#ifdef GPU_DEVICE_Pascal
// specifics see: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
// register file size 64k 32-bit registers per SM
// shared memory 64KB for GP100 and 96KB for GP104
#undef USE_LAUNCH_BOUNDS
#endif

#ifdef GPU_DEVICE_Volta
// specifics see: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
// register file size 64k 32-bit registers per SM
// shared memory size 96KB per SM (maximum shared memory per thread block)
// maximum registers 255 per thread
#undef USE_LAUNCH_BOUNDS
#endif

#ifdef GPU_DEVICE_Turing
// specifics see: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
// register file size 64k 32-bit registers per SM
// shared memory size 64KB per SM (maximum shared memory per thread block)
// maximum registers 255 per thread
#undef USE_LAUNCH_BOUNDS
#endif

#ifdef GPU_DEVICE_Ampere
// specifics see: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
// register file size 64k 32-bit registers per SM
// shared memory size 164KB per SM (maximum shared memory, 163KB per thread block)
// maximum registers 255 per thread
#undef USE_LAUNCH_BOUNDS
#endif

// acoustic kernel
// performance statistics: kernel Kernel_2_acoustic_impl():
//       shared memory per block = 2200    for Kepler: -> limits active blocks to 16 (maximum possible)
//       registers per thread    = 40
//       registers per block     = 5120                -> limits active blocks to 12
// note: for K20x, using a minimum of 16 blocks leads to register spilling.
//       this slows down the kernel by ~ 4%
#define LAUNCH_MIN_BLOCKS_ACOUSTIC 16

/* ----------------------------------------------------------------------------------------------- */

// cuda kernel block size for updating displacements/potential (newmark time scheme)
// current hardware: 128 is slightly faster than 256 ( ~ 4%)
#define BLOCKSIZE1 128
#define BLOCKSIZE2x 16
#define BLOCKSIZE2y1 8
#define BLOCKSIZE2y2 16
#define BLOCKSIZE3 128 
// maximum grid dimension in one direction of GPU
#define MAXIMUM_GRID_DIM 65535

/*----------------------------------------------------------------------------------------------- */

// balancing work group x/y-size
#undef BALANCE_WORK_GROUP

#ifdef BALANCE_WORK_GROUP
#pragma message ("\nCompiling with: BALANCE_WORK_GROUP enabled\n")
// maximum number of work group units in one dimension
#define BALANCE_WORK_GROUP_UNITS 7552 // == 32 * 236 for Knights Corner test
#endif

/* ----------------------------------------------------------------------------------------------- */

// indexing
#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + xsize*(y + ysize*z)
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*(z + zsize*i))
#define INDEX5(xsize,ysize,zsize,isize,x,y,z,i,j) x + xsize*(y + ysize*(z + zsize*(i + isize*(j))))
#define INDEX6(xsize,ysize,zsize,isize,jsize,x,y,z,i,j,k) x + xsize*(y + ysize*(z + zsize*(i + isize*(j + jsize*k))))

#define INDEX4_PADDED(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*z) + (i)*NGLL3_PADDED

/*----------------------------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------------------------------- */

// custom type declarations

/* ----------------------------------------------------------------------------------------------- */

// textures
//typedef texture<float, cudaTextureType1D, cudaReadModeElementType> realw_texture;


/*------------------------------------------------------------------------------------------------*/
//				use read only cache to cache data from global  memory
/*-------------------------------------------------------------------------------------------------*/
//	Compute Capability 3.x
//			L1 cache is used for local memory, L2 cache is used for global mem
//			Each SM has read-only cache of 48 KB, which is faster than L2, and slower than L1
//			L1 and shared memory are sharing the same on-chip memory (setup in runtime)
//	Compute capability 5.x
//			L1/texture cache of 24KB to cache read only global mem
//			64KB / 96KB shared mem
//			L2 cache for local and globale mem
//	  

// pointer declarations
// restricted pointers: may improve performance on Kepler
//	The effects here are a reduced number of memory accesses and reduced number of computations. 
//	This is balanced by an increase in register pressure due to "cached" loads and common sub-expressions.
//	Since register pressure is a critical issue in many CUDA codes, 
//	use of restricted pointers can have negative performance impact on CUDA code, due to reduced occupancy.
//	https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage
//
//   see: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict
//   however, compiler tends to use texture loads for restricted memory arrays, which might slow down performance
//
// non-restricted (default)
//typedef const realw* realw_const_p;
// restricted
typedef const realw* __restrict__ realw_const_p;
//
// non-restricted (default)
//typedef realw* realw_p;
// restricted
typedef realw* __restrict__ realw_p;

// wrapper for global memory load function
// usage:  val = get_global_cr( &A[index] );
//XXX consider further optimization using load and store functions using cache hints
//	see: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//	and also see: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
#if __CUDA_ARCH__ >= 350
// Device has ldg, read only data cache load function
__device__ __forceinline__ realw get_global_cr(realw_const_p ptr) { return __ldg(ptr); }
#else
//Device does not, fall back.
__device__ __forceinline__ realw get_global_cr(realw_const_p ptr) { return (*ptr); }
#endif

#ifdef USE_SINGLE_PRECISION
__device__ __forceinline__ realw MyPow(realw x, int a) {return _powf(x,a);}
__device__ __forceinline__ realw MyPow(realw x, realw a) {return _powf(x,a);}

typedef float2 field;
inline __host__ __device__ field Make_field(realw* b){ return make_float2(b[0],b[1]);}
inline __host__ __device__ field Make_field(realw a, realw b){ return make_float2(a,b);}
inline __host__ __device__ field Make_field(realw b){ return make_float2(b,b);}
inline __host__ __device__ realw fabs(field b){ return max(fabs(b.x),fabs(b.y));}
inline __host__ __device__ void operator+=(field &a, field b){a.x += b.x;a.y += b.y;}
inline __host__ __device__ void operator-=(field &a, field b){a.x -= b.x;a.y -= b.y;}
inline __host__ __device__ field operator*(field a, realw b){ return make_float2(a.x * b, a.y * b);}
inline __host__ __device__ field operator*(realw b, field a){ return make_float2(a.x * b, a.y * b);}
inline __host__ __device__ field operator*(field a, field b){ return make_float2(a.x * b.x, a.y * b.y);}
inline __host__ __device__ field operator+(field a, field b){ return make_float2(a.x + b.x, a.y + b.y);}
inline __host__ __device__ field operator/(field a, realw b){ return make_float2(a.x / b, a.y / b);}
inline __host__ __device__ field operator-(field a, field b){ return make_float2(a.x - b.x, a.y - b.y);}
inline __host__ __device__ field operator-(field a){ return make_float2(-a.x,-a.y);}
inline __host__ __device__ field iHilbert(field a){ return make_float2(-a.y,a.x);}
inline __host__ __device__ realw sum(field b){ return b.x+b.y;}
inline __host__ __device__ realw diff(field b){ return b.x-b.y;}
inline __host__ __device__ realw ave(field b){ return 0.5f*sum(b);}
inline __device__ void atomicAdd(field* address, field val){atomicAdd(&(address->x),val.x); atomicAdd(&(address->y),val.y);}

//dummy overloads, just to enable compilation
inline __host__ __device__ field operator+(field a, realw b){ return a;}
inline __device__ void atomicAdd(field* address, float val){}
inline __device__ void atomicAdd(realw* address, field val){}
inline __host__ __device__ void operator+=(realw &a, field b){}
// work-around to have something like: realw a = realw_(field)
inline __host__ __device__ realw realw_(field b){ return b.x; }

//positions
typedef float3 Pxyz;
inline __host__ __device__ Pxyz Make_Pxyz(realw* b){ return make_float3(b[0],b[1],b[2]);}
inline __host__ __device__ Pxyz Make_Pxyz(realw a, realw b, realw c){ return make_float3(a,b,c);}
inline __host__ __device__ Pxyz Make_Pxyz(realw b){ return make_float3(b,b,b);}
inline __host__ __device__ realw fabs(Pxyz b){ return max((max(fabs(b.x),fabs(b.y))),fabs(b,z));}
inline __host__ __device__ void operator+=(Pxyz &a, Pxyz b){a.x += b.x;a.y += b.y;a.z += b.z;}
inline __host__ __device__ void operator-=(Pxyz &a, Pxyz b){a.x -= b.x;a.y -= b.y;a.z -= b.z;}
inline __host__ __device__ Pxyz operator*(Pxyz a, realw b){ return make_float3(a.x * b, a.y * b, a.z * b);}
inline __host__ __device__ Pxyz operator*(realw b, Pxyz a){ return make_float3(a.x * b, a.y * b, a.z * b);}
inline __host__ __device__ Pxyz operator*(Pxyz a, Pxyz b){ return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);}
inline __host__ __device__ Pxyz operator+(Pxyz a, Pxyz b){ return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);}
inline __host__ __device__ Pxyz operator/(Pxyz a, realw b){ return make_float3(a.x / b, a.y / b, a.z / b);}
inline __host__ __device__ Pxyz operator-(Pxyz a, Pxyz b){ return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}
inline __host__ __device__ Pxyz operator-(Pxyz a){ return make_float3(-a.x,-a.y,-a.z);} 

#else
__device__ __forceinline__ realw MyPow(realw x, int a) {return pow(x,a);}
__device__ __forceinline__ realw MyPow(realw x, realw a) {return pow(x,a);}

typedef double2 field;
inline __host__ __device__ field Make_field(realw* b){ return make_double2(b[0],b[1]);}
inline __host__ __device__ field Make_field(realw a, realw b){ return make_double2(a,b);}
inline __host__ __device__ field Make_field(realw b){ return make_double2(b,b);}
inline __host__ __device__ realw fabs(field b){ return max(abs(b.x),abs(b.y));}
inline __host__ __device__ void operator+=(field &a, field b){a.x += b.x;a.y += b.y;}
inline __host__ __device__ void operator-=(field &a, field b){a.x -= b.x;a.y -= b.y;}
inline __host__ __device__ field operator*(field a, realw b){ return make_double2(a.x * b, a.y * b);}
inline __host__ __device__ field operator*(realw b, field a){ return make_double2(a.x * b, a.y * b);}
inline __host__ __device__ field operator*(field a, field b){ return make_double2(a.x * b.x, a.y * b.y);}
inline __host__ __device__ field operator+(field a, field b){ return make_double2(a.x + b.x, a.y + b.y);}
inline __host__ __device__ field operator/(field a, realw b){ return make_double2(a.x / b, a.y / b);}
inline __host__ __device__ field operator-(field a, field b){ return make_double2(a.x - b.x, a.y - b.y);}
inline __host__ __device__ field operator-(field a){ return make_double2(-a.x,-a.y);}
inline __host__ __device__ field iHilbert(field a){ return make_double2(-a.y,a.x);}
inline __host__ __device__ realw sum(field b){ return b.x+b.y;}
inline __host__ __device__ realw diff(field b){ return b.x-b.y;}
inline __host__ __device__ realw ave(field b){ return 0.5f*sum(b);}
// atomicAdd do not support double prcession for architecture older than sm_60
//inline __device__ void atomicAdd(field* address, field val){atomicAdd(&(address->x),val.x); atomicAdd(&(address->y),val.y);}

//dummy overloads, just to enable compilation
inline __host__ __device__ field operator+(field a, realw b){ return a;}
//inline __device__ void atomicAdd(field* address, float val){}
//inline __device__ void atomicAdd(realw* address, field val){}
inline __host__ __device__ void operator+=(realw &a, field b){}
// work-around to have something like: realw a = realw_(field)
inline __host__ __device__ realw realw_(field b){ return b.x; }

typedef double3 Pxyz;
inline __host__ __device__ Pxyz Make_Pxyz(realw* b){ return make_double3(b[0],b[1],b[2]);}
inline __host__ __device__ Pxyz Make_Pxyz(realw a, realw b, realw c){ return make_double3(a,b,c);}
inline __host__ __device__ Pxyz Make_Pxyz(realw b){ return make_double3(b,b,b);}
inline __host__ __device__ realw fabs(Pxyz b){ return max((max(abs(b.x),abs(b.y))),abs(b.z));}
inline __host__ __device__ void operator+=(Pxyz &a, Pxyz b){a.x += b.x;a.y += b.y;a.z += b.z;}
inline __host__ __device__ void operator-=(Pxyz &a, Pxyz b){a.x -= b.x;a.y -= b.y;a.z -= b.z;}
inline __host__ __device__ Pxyz operator*(Pxyz a, realw b){ return make_double3(a.x * b, a.y * b, a.z * b);}
inline __host__ __device__ Pxyz operator*(realw b, Pxyz a){ return make_double3(a.x * b, a.y * b, a.z * b);}
inline __host__ __device__ Pxyz operator*(Pxyz a, Pxyz b){ return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);}
inline __host__ __device__ Pxyz operator+(Pxyz a, Pxyz b){ return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);}
inline __host__ __device__ Pxyz operator/(Pxyz a, realw b){ return make_double3(a.x / b, a.y / b, a.z / b);}
inline __host__ __device__ Pxyz operator-(Pxyz a, Pxyz b){ return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);}
inline __host__ __device__ Pxyz operator-(Pxyz a){ return make_double3(-a.x,-a.y,-a.z);}

#endif


#ifdef USE_SINGLE_PRECISION
#define zerof	0.0f
#else
#define zerof 0.0
#endif

/* ----------------------------------------------------------------------------------------------- */

// utility functions

/* ----------------------------------------------------------------------------------------------- */

// defined in check_fields_cuda.cu
double get_time_val();
void get_free_memory(double* free_db, double* used_db, double* total_db);
void print_CUDA_error_if_any(cudaError_t err, int num);
void pause_for_debugger(int pause);
void print_cuFFT_error_if_any(cufftResult err, int num);
void print_cuSparse_error_if_any(cusparseStatus_t err, int num);

void exit_on_cuda_error(const char* kernel_name);
void exit_on_error(const char* info);

void synchronize_cuda();
void synchronize_mpi();

void start_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop);
void stop_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop, const char* info_str);
void stop_timing_cuda(cudaEvent_t* start,cudaEvent_t* stop, const char* info_str,realw* t);

realw get_device_array_maximum_value(realw* array,int size);

// defined in helper_functions.cu
void copy_todevice_int(void** d_array_addr_ptr,int* h_array,int size);
void copy_todevice_realw(void** d_array_addr_ptr,realw* h_array,int size);

// defined in initialize_cuda_device.cu
void initialize_cuda_device(int myrank,int* ncuda_devices); 

/* ----------------------------------------------------------------------------------------------- */

// kernel setup function

/* ----------------------------------------------------------------------------------------------- */

// moved here into header to inline function calls if possible

static inline void get_blocks_xy(int num_blocks, int* num_blocks_x, int* num_blocks_y) {

// Initially sets the blocks_x to be the num_blocks, and adds rows as needed (block size limit of 65535).
// If an additional row is added, the row length is cut in
// half. If the block count is odd, there will be 1 too many blocks,
// which must be managed at runtime with an if statement.

  *num_blocks_x = num_blocks;
  *num_blocks_y = 1;

  while (*num_blocks_x > MAXIMUM_GRID_DIM) {
    *num_blocks_x = (int) ceil(*num_blocks_x * 0.5f);
    *num_blocks_y = *num_blocks_y * 2;
  }

#if DEBUG == 1
  printf("work group - total %d has group size x = %d / y = %d\n",
         num_blocks,*num_blocks_x,*num_blocks_y);
#endif

  // tries to balance x- and y-group
#ifdef BALANCE_WORK_GROUP
  if (*num_blocks_x > BALANCE_WORK_GROUP_UNITS && *num_blocks_y < BALANCE_WORK_GROUP_UNITS){
    while (*num_blocks_x > BALANCE_WORK_GROUP_UNITS && *num_blocks_y < BALANCE_WORK_GROUP_UNITS) {
      *num_blocks_x = (int) ceil (*num_blocks_x * 0.5f);
      *num_blocks_y = *num_blocks_y * 2;
    }
  }

#if DEBUG == 1
  printf("balancing work group with limit size %d - total %d has group size x = %d / y = %d\n",
         BALANCE_WORK_GROUP_UNITS,num_blocks,*num_blocks_x,*num_blocks_y);
#endif

#endif
}


#endif  // CUDA_DEVICE_H
