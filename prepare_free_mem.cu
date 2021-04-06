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
   It is modified from 'prepare_mesh_constants_cuda.h' from the program
   'specfem3d version 3.0' at https://github.com/geodynamics/specfem3d
*/

#include "Cuda_device.h"
#include"FTDT.h"
#include<math.h>
/* ----------------------------------------------------------------------------------------------- */

// GPU preparation

/* ----------------------------------------------------------------------------------------------- */
  
void prepare_device_arrays(DArrays* DPtrs, ModelPara MP, realw ***HostCond, 
													 RxPos *d_xyzRx, RxPos xyzRx, int NRx, int **RxLst,
													 ModelGrid *d_MG, ModelGrid MG, GridConv *d_GC, GridConv GC)
{

  TRACE("prepare device arrays");
	int L,M,N,L1,M1;
	
	L=MP.L;
	M=MP.M;
	N=MP.N;
	
	L1=MP.AnLx_New+MP.AnRx_New+MP.Nx;
  M1=2*MP.Any_New+MP.Ny;
#if MAXDEBUG == 1
	printf("L=%d, M=%d, N=%d\n",L,M,N);
#endif				
	// EM field
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Ex), L*(M+1)*(N+1)*sizeof(realw)), 1001);
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Ey), (L+1)*(M)*(N+1)*sizeof(realw)), 1002);
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Ez), (L+1)*(M+1)*(N)*sizeof(realw)), 1003);
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Bx), (L+1)*(M)*(N)*sizeof(realw)), 1004);
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->By), L*(M+1)*(N)*sizeof(realw)), 1005);
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Bz), L*(M)*(N+1)*sizeof(realw)), 1006);
	
	// Mem for Bx air and By air, 2D padded Mem
	// XXX may consider using texture mem in a future version

	print_CUDA_error_if_any( cudaMallocPitch(&(DPtrs->BxAir), &(DPtrs->BxPitch), (L+1)*sizeof(realw), M), 1007);
	print_CUDA_error_if_any( cudaMallocPitch(&(DPtrs->ByAir), &(DPtrs->ByPitch), (L)*sizeof(realw), M+1), 1008);	
	
	TRACE("Copy conductivity model from host to device\n");
	//conductivity
	print_CUDA_error_if_any( cudaMalloc((void **)&(DPtrs->Con), L*(M)*(N)*sizeof(realw)), 1011);
	print_CUDA_error_if_any( cudaMemcpy(DPtrs->Con, HostCond[0][0], L*(M)*(N)*sizeof(realw), cudaMemcpyHostToDevice), 1012 );
	// free host conductivity 
	Free_3D_Array(HostCond,N);
	
	//Structure of arrays
	// https://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays
	TRACE("Copy receiver arrays from host to device\n");
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->x), NRx*sizeof(realw)), 1013);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->x, xyzRx.x, NRx*sizeof(realw), cudaMemcpyHostToDevice), 1014 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->y), NRx*sizeof(realw)), 1015);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->y, xyzRx.y, NRx*sizeof(realw), cudaMemcpyHostToDevice), 1016 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->z), NRx*sizeof(realw)), 1017);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->z, xyzRx.z, NRx*sizeof(realw), cudaMemcpyHostToDevice), 1018 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->ix), NRx*sizeof(int)), 1019);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->ix, xyzRx.ix, NRx*sizeof(int), cudaMemcpyHostToDevice), 1020 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->iy), NRx*sizeof(int)), 1021);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->iy, xyzRx.iy, NRx*sizeof(int), cudaMemcpyHostToDevice), 1022 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_xyzRx->iz), NRx*sizeof(int)), 1023);
	print_CUDA_error_if_any( cudaMemcpy(d_xyzRx->iz, xyzRx.iz, NRx*sizeof(int), cudaMemcpyHostToDevice), 1024 );
	
	
	// allocate RxLst on device
	print_CUDA_error_if_any( cudaMalloc((void **)RxLst, NRx*sizeof(int)), 1025);
	
	TRACE("Allocate model grid arrays on device\n");
	// allocate Model Grid arrays
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->dx), L*sizeof(realw)), 1027);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->dx, MG.dx, L*sizeof(realw), cudaMemcpyHostToDevice), 1028 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->dy), M*sizeof(realw)), 1029);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->dy, MG.dy, M*sizeof(realw), cudaMemcpyHostToDevice), 1030 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->dz), N*sizeof(realw)), 1031);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->dz, MG.dz, N*sizeof(realw), cudaMemcpyHostToDevice), 1032 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->X_Bzold), L*sizeof(realw)), 1033);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->X_Bzold, MG.X_Bzold, L*sizeof(realw), cudaMemcpyHostToDevice), 1034 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->Y_Bzold), M*sizeof(realw)), 1035);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->Y_Bzold, MG.Y_Bzold, M*sizeof(realw), cudaMemcpyHostToDevice), 1036 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->Z), N*sizeof(realw)), 1037);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->Z, MG.Z, N*sizeof(realw), cudaMemcpyHostToDevice), 1038 );	
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->X_BzNew), (MP.AnLx_New+MP.AnRx_New+MP.Nx)*sizeof(realw)), 1039);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->X_BzNew, MG.X_BzNew, (MP.AnLx_New+MP.AnRx_New+MP.Nx)*sizeof(realw), cudaMemcpyHostToDevice), 1040 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_MG->Y_BzNew), (2*MP.Any_New+MP.Ny)*sizeof(realw)), 1041);
	print_CUDA_error_if_any( cudaMemcpy(d_MG->Y_BzNew, MG.Y_BzNew, (2*MP.Any_New+MP.Ny)*sizeof(realw), cudaMemcpyHostToDevice), 1042 );
	
	
	//alocate grid mapping arrays
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->xBzNew2old), L1*sizeof(int)), 1201);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->xBzNew2old, GC.xBzNew2old, L1*sizeof(int), cudaMemcpyHostToDevice), 1202 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->yBzNew2old), M1*sizeof(int)), 1203);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->yBzNew2old, GC.yBzNew2old, M1*sizeof(int), cudaMemcpyHostToDevice), 1204 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->xBzold2New), L*sizeof(int)), 1205);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->xBzold2New, GC.xBzold2New, L*sizeof(int), cudaMemcpyHostToDevice), 1206 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->yBzold2New), M*sizeof(int)), 1207);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->yBzold2New, GC.yBzold2New, M*sizeof(int), cudaMemcpyHostToDevice), 1208 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->xoldBx2NewBz), (L+1)*sizeof(int)), 1209);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->xoldBx2NewBz, GC.xoldBx2NewBz, (L+1)*sizeof(int), cudaMemcpyHostToDevice), 1210 );
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(d_GC->yoldBy2NewBz), (M+1)*sizeof(int)), 1211);
	print_CUDA_error_if_any( cudaMemcpy(d_GC->yoldBy2NewBz, GC.yoldBy2NewBz, (M+1)*sizeof(int), cudaMemcpyHostToDevice), 1212 );
	
  GPU_ERROR_CHECKING("prepare device arrays");
}


/* ----------------------------------------------------------------------------------------------- */

// set device memory to zeros

/* ----------------------------------------------------------------------------------------------- */
void set_zeros_EM_arrays(DArrays* DPtrs, ModelPara *MP)
{

  TRACE("set EM field (device arrays) to zeros");
	int L,M,N;
	
	L=MP->L;
	M=MP->M;
	N=MP->N;
				
	// EM field
	print_CUDA_error_if_any( cudaMemset( DPtrs->Ex, 0, L*(M+1)*(N+1)*sizeof(realw)), 1101);
	print_CUDA_error_if_any( cudaMemset( DPtrs->Ey, 0, (L+1)*(M)*(N+1)*sizeof(realw)), 1102);
	print_CUDA_error_if_any( cudaMemset( DPtrs->Ez, 0, (L+1)*(M+1)*(N)*sizeof(realw)), 1103);
	print_CUDA_error_if_any( cudaMemset( DPtrs->Bx, 0, (L+1)*(M)*(N)*sizeof(realw)), 1104);
	print_CUDA_error_if_any( cudaMemset( DPtrs->By, 0, (L)*(M+1)*(N)*sizeof(realw)), 1105);
	print_CUDA_error_if_any( cudaMemset( DPtrs->Bz, 0, (L)*(M)*(N+1)*sizeof(realw)), 1106);
		
	print_CUDA_error_if_any( cudaMemset2D((DPtrs->BxAir), (DPtrs->BxPitch), 0, (L+1)*sizeof(realw), M), 1107);
	print_CUDA_error_if_any( cudaMemset2D((DPtrs->ByAir), (DPtrs->ByPitch), 0, (L)*sizeof(realw), M+1), 1008);	
	
	GPU_ERROR_CHECKING("set EM field (device arrays) to zeros");
}



/* ----------------------------------------------------------------------------------------------- */
// cleanup
/* ----------------------------------------------------------------------------------------------- */

void Device_cleanup(DArrays* DPtrs, RxPos *d_xyzRx, int **RxLst, ModelGrid *d_MG, GridConv *d_GC) 
{

	TRACE("Cleanup_device");

  // frees memory on GPU
  print_CUDA_error_if_any( cudaFree( (DPtrs->Ex) ), 3101);
	print_CUDA_error_if_any( cudaFree( (DPtrs->Ey) ), 3102);
	print_CUDA_error_if_any( cudaFree( (DPtrs->Ez) ), 3103);
	print_CUDA_error_if_any( cudaFree( (DPtrs->Bx) ), 3104);
	print_CUDA_error_if_any( cudaFree( (DPtrs->By) ), 3105);
	print_CUDA_error_if_any( cudaFree( (DPtrs->Bz) ), 3106);
	print_CUDA_error_if_any( cudaFree( DPtrs->BxAir ), 3107);
	print_CUDA_error_if_any( cudaFree( DPtrs->ByAir ), 3108);
	
	print_CUDA_error_if_any( cudaFree( (DPtrs->Con) ), 3109);
	
	print_CUDA_error_if_any( cudaFree( *RxLst ), 3110);
	
	print_CUDA_error_if_any( cudaFree( d_xyzRx->x ), 3111);
	print_CUDA_error_if_any( cudaFree( d_xyzRx->y ), 3112);
	print_CUDA_error_if_any( cudaFree( d_xyzRx->z ), 3113);
	
	print_CUDA_error_if_any( cudaFree( d_xyzRx->ix ), 3114);
	print_CUDA_error_if_any( cudaFree( d_xyzRx->iy ), 3115);
	print_CUDA_error_if_any( cudaFree( d_xyzRx->iz ), 3116);
	
	print_CUDA_error_if_any( cudaFree( d_MG->dx ), 3117);
	print_CUDA_error_if_any( cudaFree( d_MG->dy ), 3118);
	print_CUDA_error_if_any( cudaFree( d_MG->dz ), 3119);
	
	print_CUDA_error_if_any( cudaFree( d_MG->X_Bzold ), 3120);
	print_CUDA_error_if_any( cudaFree( d_MG->Y_Bzold ), 3121);
	print_CUDA_error_if_any( cudaFree( d_MG->Z ), 3122);
	print_CUDA_error_if_any( cudaFree( d_MG->X_BzNew ), 3123);
	print_CUDA_error_if_any( cudaFree( d_MG->Y_BzNew ), 3124);
	
	print_CUDA_error_if_any( cudaFree( d_GC->xBzNew2old ), 3125);
	print_CUDA_error_if_any( cudaFree( d_GC->yBzNew2old ), 3126);
	print_CUDA_error_if_any( cudaFree( d_GC->xBzold2New ), 3127);
	print_CUDA_error_if_any( cudaFree( d_GC->yBzold2New ), 3128);
	print_CUDA_error_if_any( cudaFree( d_GC->xoldBx2NewBz ), 3129);
	print_CUDA_error_if_any( cudaFree( d_GC->yoldBy2NewBz ), 3130);
	
	GPU_ERROR_CHECKING("Clean arrays on device");	
}

//cpy data to host for fftw
#ifdef FFTW
void CP2Host_bz0_UPP(DArrays* DPtrs, realw *bz0, int M, int L)
{
	TRACE("cpy top slice of Bz to host to do upward continuation");
	
	print_CUDA_error_if_any( cudaMemcpy( bz0, DPtrs->Bz, L*M*sizeof(realw), cudaMemcpyDeviceToHost), 1201);
	
	GPU_ERROR_CHECKING("cpy top slice to host");
#if MAXDEBUG == 1
	for(int j=0; j<M; j++)
	{
		for(int i=0; i<L; i++)
		{
			if( isnan(bz0[j*M+i]) )
			{
				printf("Error in bz0 j=%d,i=%d\n",j,i);
				pause();
			}
		}
	}
#endif	
}

void CP2Device_bxby_UPP(DArrays* DPtrs, realw *bx0, realw *by0, int M, int L)
{
	TRACE("cpy BxAir and ByAir to device");
#if MAXDEBUG == 1
	for(int j=0; j<M; j++)
	{
		for(int i=0; i<L; i++)
		{
			if( isnan(bx0[j*M+i]) || isnan(by0[j*(M+1)+i]) )
			{
				printf("Error in BxAir or ByAir j=%d,i=%d\n",j,i);
				pause();
			}
		}
	}
#endif	
	print_CUDA_error_if_any( cudaMemcpy2D( DPtrs->BxAir, (DPtrs->BxPitch), bx0, (L+1)*sizeof(realw), 
													 (L+1)*sizeof(realw), M, cudaMemcpyHostToDevice), 1202);
	
	print_CUDA_error_if_any( cudaMemcpy2D( DPtrs->ByAir, (DPtrs->ByPitch), by0, (L)*sizeof(realw), 
													 (L)*sizeof(realw), M+1, cudaMemcpyHostToDevice), 1203);		
													 										 
	GPU_ERROR_CHECKING("cpy BxAir ByAir to device");												 
}
#endif

