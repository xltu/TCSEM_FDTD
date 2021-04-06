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
 // REVIEW Corrections and Updates
 //	XT @ 2021.02.25 Instead of padding only two points as in the interpolation in the previous
 // 								version 20210216, this version padding more (NPad) points in the boundary.
 //									 This is only for accuracy consideration. Seems not too much improvements
 //
 // XT @ 2021.02.26 Reverse the direction of y direction interpolation equation system for the back
 //									part of the interpolation to improve tridiagonal solver accuracy		
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<unistd.h>
#include "FTDT.h"

#include<cuda.h>
#include<cuda_runtime.h>
#include "Cuda_device.h"
/*
*	Need a way to check whether A is right	
*/
__global__ void fillA0x( TriDiagM A, const ModelPara *d_mp, ModelGrid MG )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int n1= d_mp->ALx1 + d_mp->ALx2 + NPad;
	if( ix < n1 )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		sh_dx[threadIdx.x] = MG.dx[threadIdx.x];
		__syncthreads( );

		if( ix == 0 || ix == (n1-1) )
		{
			A.d[ix]=2.0;
			A.dl[ix]=zerof;
			A.du[ix]=zerof;	
		}	
		else 
		{
			A.d[ix] = 2.0/3.0*(sh_dx[ix]+0.5*sh_dx[ix-1]+0.5*sh_dx[ix+1]);
			A.dl[ix] = 1.0/3.0*(0.5*sh_dx[ix-1]+0.5*sh_dx[ix]);							// A(i,i-1)
			A.du[ix] = 1.0/3.0*(0.5*sh_dx[ix]+0.5*sh_dx[ix+1]);							// A(i,i+1)
		}
	}	
}	

__global__ void fillA1x( TriDiagM A, const ModelPara *d_mp, ModelGrid MG )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int n2=d_mp->ARx1 + d_mp->ARx2 + NPad;
	if( ix < n2 )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		sh_dx[threadIdx.x] = MG.dx[d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + threadIdx.x];
		__syncthreads( );

		if( ix == 0 || ix == (n2-1) )
		{
			A.d[ix]=2.0;
			A.dl[ix]=zerof;
			A.du[ix]=zerof;	
		}	
		else 
		{
			A.d[ix] = 2.0/3.0*( sh_dx[ix]+0.5*sh_dx[ix-1]+0.5*sh_dx[ix+1] );
			A.dl[ix] = 1.0/3.0*( 0.5*sh_dx[ix-1] + 0.5*sh_dx[ix] );
			A.du[ix] = 1.0/3.0*( 0.5*sh_dx[ix] + 0.5*sh_dx[ix+1] );
		}
	}	
}	

__global__ void fillA0y( TriDiagM A, const ModelPara *d_mp, ModelGrid MG )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2 + NPad;
	if( ix < n3 )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		sh_dy[threadIdx.x] = MG.dy[threadIdx.x];
		__syncthreads( );

		if( ix == 0 || ix == (n3-1) )
		{
			A.d[ix]=2.0;	
			A.dl[ix]=zerof;
			A.du[ix]=zerof;
		}	
		else 
		{
			A.d[ix] = 2.0/3.0*(sh_dy[ix]+0.5*sh_dy[ix-1]+0.5*sh_dy[ix+1]);
			A.dl[ix] = 1.0/3.0*(0.5*sh_dy[ix-1]+0.5*sh_dy[ix]);
			A.du[ix] = 1.0/3.0*(0.5*sh_dy[ix]+0.5*sh_dy[ix+1]);
		}
	}	
}	

__global__ void fillA1y( TriDiagM A, const ModelPara *d_mp, ModelGrid MG )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2 + NPad;
	if( ix < n3 )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		sh_dy[threadIdx.x] = MG.dy[d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + threadIdx.x];
		__syncthreads( );

		if( ix == 0 || ix == (n3-1) )
		{
			A.d[ix]=2.0;
			A.dl[ix]=zerof;
			A.du[ix]=zerof;	
		}	
		else 
		{
			A.d[ix] = 2.0/3.0*(sh_dy[ix]+0.5*sh_dy[ix-1]+0.5*sh_dy[ix+1]);
			A.dl[ix] = 1.0/3.0*(0.5*sh_dy[ix-1]+0.5*sh_dy[ix]);
			A.du[ix] = 1.0/3.0*(0.5*sh_dy[ix]+0.5*sh_dy[ix+1]);
		}
	}	
}	

void initialize_UpCont_gpu_resources(UpContP *UPP, ModelPara *MP, ModelPara *d_mp, ModelGrid *d_MG, cudaStream_t *stream)
{
	TRACE("initialize the Upward continuation arrays and gpu handles on GPU");
	initialize_fft(UPP, MP, stream);

	int M, L, M1, L1, irxL_old, irxR_old, iry_old;
	irxL_old = MP->ALx1 + MP->ALx2;		// to keep name consistant with version 1.0 code
	irxR_old = MP->ARx1 + MP->ARx2;	
	iry_old = MP->Ay1 + MP->Ay2;

	L1=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M1=2*MP->Any_New+MP->Ny;

  L=MP->L;
  M=MP->M;

	if( (irxL_old + NPad)> BLOCKSIZE3 || (irxR_old + NPad)> BLOCKSIZE3 || (iry_old + NPad)> BLOCKSIZE3 )
		exit_on_error("BLOCKSIZE3 (l222 in 'Cuda_device.h') is too small, it should be larger than AnLx1 + AnLx2 + NPad (AnRx1 + AnRx2 + NPad)");
	//  Allocate global mem on GPU
	// x direction interpolation in the left
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).A.dl ), 	(irxL_old + NPad)*sizeof(realw)), 41101);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).A.du ), 	(irxL_old + NPad)*sizeof(realw)), 41102);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).A.d ), 	(irxL_old + NPad)*sizeof(realw)), 41103);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).B ), 	M*(irxL_old + NPad)*sizeof(realw)), 41104);	// XXX M is not a error

	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0x).A.dl, 0, (irxL_old + NPad)*sizeof(realw)), 41115);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0x).A.du, 0, (irxL_old + NPad)*sizeof(realw)), 41116);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0x).B, 0, M*(irxL_old + NPad)*sizeof(realw)), 41117);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).bz_semi0 ), 	MP->AnLx_New*(iry_old + NPad)*sizeof(realw)), 41121);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).bz_semi1 ), 	MP->AnLx_New*(iry_old + NPad)*sizeof(realw)), 41122);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0x).bz_semi0, 0, MP->AnLx_New*(iry_old + NPad)*sizeof(realw)), 41123);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0x).bz_semi1, 0, MP->AnLx_New*(iry_old + NPad)*sizeof(realw)), 41124);	

	print_cuSparse_error_if_any( cusparseCreate(&( (UPP->TDS0x).cuSPHandle)), 41105);

	// x direction interpolation in the right
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).A.dl ), 	(irxR_old + NPad)*sizeof(realw)), 41201);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).A.du ), 	(irxR_old + NPad)*sizeof(realw)), 41202);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).A.d ), 	(irxR_old + NPad)*sizeof(realw)), 41203);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).B ), 	M*(irxR_old + NPad)*sizeof(realw)), 41204); // XXX M is not a error

	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1x).A.dl, 0, (irxR_old + NPad)*sizeof(realw)), 41215);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1x).A.du, 0, (irxR_old + NPad)*sizeof(realw)), 41216);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1x).B, 0, M*(irxR_old + NPad)*sizeof(realw)), 41217);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).bz_semi0 ), 	MP->AnRx_New*(iry_old + NPad)*sizeof(realw)), 41221);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).bz_semi1 ), 	MP->AnRx_New*(iry_old + NPad)*sizeof(realw)), 41222);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1x).bz_semi0, 0, MP->AnRx_New*(iry_old + NPad)*sizeof(realw)), 41223);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1x).bz_semi1, 0, MP->AnRx_New*(iry_old + NPad)*sizeof(realw)), 41224);	

	print_cuSparse_error_if_any( cusparseCreate(&( (UPP->TDS1x).cuSPHandle)), 41205);

	// y direction interpolation in the front
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0y).A.dl ), 	(iry_old + NPad)*sizeof(realw)), 41301);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0y).A.du ), 	(iry_old + NPad)*sizeof(realw)), 41302);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0y).A.d ), 	(iry_old + NPad)*sizeof(realw)), 41303);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0y).B ), 	L1*(iry_old + NPad)*sizeof(realw)), 41304);

	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0y).A.dl, 0, (iry_old + NPad)*sizeof(realw)), 41315);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0y).A.du, 0, (iry_old + NPad)*sizeof(realw)), 41316);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS0y).B, 0, L1*(iry_old + NPad)*sizeof(realw)), 41317);

	print_cuSparse_error_if_any( cusparseCreate(&( (UPP->TDS0y).cuSPHandle)), 41305);

	// y directiion interpolation in the back
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1y).A.dl ), 	(iry_old + NPad)*sizeof(realw)), 41401);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1y).A.du ), 	(iry_old + NPad)*sizeof(realw)), 41402);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1y).A.d ), 	(iry_old + NPad)*sizeof(realw)), 41403);

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1y).B ), 	L1*(iry_old + NPad)*sizeof(realw)), 41404);

	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1y).A.dl, 0, (iry_old + NPad)*sizeof(realw)), 41415);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1y).A.du, 0, (iry_old + NPad)*sizeof(realw)), 41416);
	print_CUDA_error_if_any( cudaMemset( (UPP->TDS1y).B, 0, L1*(iry_old + NPad)*sizeof(realw)), 41417);

	print_cuSparse_error_if_any( cusparseCreate(&( (UPP->TDS1y).cuSPHandle)), 41405);	

	//set the streams
	print_cuSparse_error_if_any( cusparseSetStream( (UPP->TDS0x).cuSPHandle , stream[0] ), 510001);
	print_cuSparse_error_if_any( cusparseSetStream( (UPP->TDS1x).cuSPHandle , stream[1] ), 510002);
	print_cuSparse_error_if_any( cusparseSetStream( (UPP->TDS0y).cuSPHandle , stream[0] ), 510003);
	print_cuSparse_error_if_any( cusparseSetStream( (UPP->TDS1y).cuSPHandle , stream[1] ), 510004);

	// allocate the buffer 
	size_t buffSize1, buffSize2, buffSize3, buffSize4;
#ifdef USE_SINGLE_PRECISION	
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot_bufferSizeExt( (UPP->TDS0x).cuSPHandle, irxL_old + NPad, M, 
																(UPP->TDS0x).A.dl, (UPP->TDS0x).A.d, (UPP->TDS0x).A.du, (UPP->TDS0x).B,
																irxL_old + NPad,  &buffSize1 ), 41106 );

	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot_bufferSizeExt( (UPP->TDS1x).cuSPHandle, irxR_old + NPad, M, 
																(UPP->TDS1x).A.dl, (UPP->TDS1x).A.d, (UPP->TDS1x).A.du, (UPP->TDS1x).B,
																irxR_old + NPad,  &buffSize2 ), 41206 );

	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot_bufferSizeExt( (UPP->TDS0y).cuSPHandle, iry_old + NPad, L1, 
																(UPP->TDS0y).A.dl, (UPP->TDS0y).A.d, (UPP->TDS0y).A.du, (UPP->TDS0y).B,
																 iry_old + NPad,  &buffSize3 ), 41306 );
																 
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot_bufferSizeExt( (UPP->TDS1y).cuSPHandle, iry_old + NPad, L1, 
																 (UPP->TDS1y).A.dl, (UPP->TDS1y).A.d, (UPP->TDS1y).A.du, (UPP->TDS1y).B,
																	iry_old + NPad,  &buffSize4 ), 41406 );															 
#else
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot_bufferSizeExt( (UPP->TDS0x).cuSPHandle, irxL_old + NPad, M, 
																(UPP->TDS0x).A.dl, (UPP->TDS0x).A.d, (UPP->TDS0x).A.du, (UPP->TDS0x).B,
																irxL_old + NPad,  &buffSize1 ), 41107 );

	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot_bufferSizeExt( (UPP->TDS1x).cuSPHandle, irxR_old + NPad, M, 
																(UPP->TDS1x).A.dl, (UPP->TDS1x).A.d, (UPP->TDS1x).A.du, (UPP->TDS1x).B,
																irxR_old + NPad,  &buffSize2 ), 41207 );

	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot_bufferSizeExt( (UPP->TDS0y).cuSPHandle, iry_old + NPad, L1, 
																(UPP->TDS0y).A.dl, (UPP->TDS0y).A.d, (UPP->TDS0y).A.du, (UPP->TDS0y).B,
																 iry_old + NPad,  &buffSize3 ), 41307 );
																 
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot_bufferSizeExt( (UPP->TDS1y).cuSPHandle, iry_old + NPad, L1, 
																 (UPP->TDS1y).A.dl, (UPP->TDS1y).A.d, (UPP->TDS1y).A.du, (UPP->TDS1y).B,
																	iry_old + NPad,  &buffSize4 ), 41407 );	
#endif	

	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0x).buff ) , buffSize1	), 41108);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1x).buff ) , buffSize2	), 41208);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS0y).buff ) , buffSize3	), 41308);
	print_CUDA_error_if_any( cudaMalloc((void **)&( (UPP->TDS1y).buff ) , buffSize4	), 41408);

	GPU_ERROR_CHECKING("initialize the Upward continuation arrays and gpu handles on GPU!");

	// filling in the A matrices
	// TODO concurrent execution of the four kernels in four stream
	TRACE("fillA0x");
	fillA0x<<<(irxL_old + NPad+BLOCKSIZE3-1)/BLOCKSIZE3, BLOCKSIZE3, 0, stream[0]>>>( (UPP->TDS0x).A, d_mp, *d_MG );
	GPU_ERROR_CHECKING("fillA0x");

	TRACE("fillA1x");
	fillA1x<<<(irxR_old + NPad+BLOCKSIZE3-1)/BLOCKSIZE3,BLOCKSIZE3, 0, stream[1]>>>( (UPP->TDS1x).A, d_mp, *d_MG );
	GPU_ERROR_CHECKING("fillA1x");	

	TRACE("fillA0y");
	fillA0y<<<(iry_old + NPad+BLOCKSIZE3-1)/BLOCKSIZE3,BLOCKSIZE3, 0, stream[2]>>>( (UPP->TDS0y).A, d_mp, *d_MG );
	GPU_ERROR_CHECKING("fillA0y");

	TRACE("fillA1y");
	fillA1y<<<(iry_old + NPad+BLOCKSIZE3-1)/BLOCKSIZE3,BLOCKSIZE3, 0, stream[3]>>>( (UPP->TDS1y).A, d_mp, *d_MG );
	GPU_ERROR_CHECKING("fillA1y");		
	
}

void release_UpCont_gpu_resources(UpContP *UPP)
{
	TRACE("release_UpCont_gpu_resources");
	destroy_fft( UPP );
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).A.dl ), 41109);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).A.d ), 41110);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).A.du ), 41111);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).B ), 41112);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).buff ), 41113);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).bz_semi0 ), 41125);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0x).bz_semi1 ), 41126);
  
	print_cuSparse_error_if_any( cusparseDestroy( (UPP->TDS0x).cuSPHandle ), 41114);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).A.dl ), 41209);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).A.d ), 41210);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).A.du ), 41211);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).B ), 41212);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).buff ), 41213);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).bz_semi0 ), 41225);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1x).bz_semi1 ), 41226);
  
	print_cuSparse_error_if_any( cusparseDestroy( (UPP->TDS1x).cuSPHandle ), 41214);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS0y).A.dl ), 41309);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0y).A.d ), 41310);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0y).A.du ), 41311);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS0y).B ), 41312);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS0y).buff ), 41313);
  
	print_cuSparse_error_if_any( cusparseDestroy( (UPP->TDS0y).cuSPHandle ), 41314);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS1y).A.dl ), 41409);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1y).A.d ), 41410);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1y).A.du ), 41411);

	print_CUDA_error_if_any( cudaFree( (UPP->TDS1y).B ), 41412);
	print_CUDA_error_if_any( cudaFree( (UPP->TDS1y).buff ), 41413);
  
	print_cuSparse_error_if_any( cusparseDestroy( (UPP->TDS1y).cuSPHandle ), 41414);

	GPU_ERROR_CHECKING("release_UpCont_gpu_resources");
}
