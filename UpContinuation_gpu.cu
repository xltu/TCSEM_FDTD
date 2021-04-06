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
/* the UpContinuation function for the DB 
*	REVIEW Corrections and Updates
*	@Xiaolei Tu @Feb.27, 2021, update the numerical form of the upward calculation equations 
*											to improve the numerical accuracy of the exponential and square root 
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <omp.h>
#include<cufft.h>
#include<cuda.h>
#include<cuda_runtime.h>

#include"FTDT.h"
#include"Cuda_device.h"

__global__ void Blinear_BxR2IR( realw *Bold, size_t BPitch, realw *Bnew,
															 const ModelPara *d_mp, ModelGrid MG, GridConv GC)
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int L0 = d_mp->L;
	const int M0 = d_mp->M;
	if(ix < L0+1 && iy < M0)
	{
		const int L1 = d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx;
		const int M1 = 2*d_mp->Any_New + d_mp->Ny;
  	
  	const unsigned int II = min( max( GC.xoldBx2NewBz[ix], 1 ), L1-1 );
  	const unsigned int JJ = min( max( GC.yBzold2New[iy], 1 ), M1-1 );
  	
  	realw X, l1, a0, a1;
  	if(ix == L0)
  		X = MG.X_Bzold[L0-1] + 0.5 * MG.dx[L0-1];
  	else
  		X = MG.X_Bzold[ix] - 0.5 * MG.dx[ix];	
  	
  	l1 = ( X - get_global_cr( &(MG.X_BzNew[II-1]) ) ) 
  			 / ( get_global_cr( &(MG.X_BzNew[II]) ) - get_global_cr( &(MG.X_BzNew[II-1]) ) );
  	
  	a0 = (1.0 - l1) * get_global_cr( & Bnew[(JJ-1)*L1+II-1] ) + l1 * get_global_cr( & Bnew[(JJ-1)*L1+II] );	
  	a1 = (1.0 - l1) * get_global_cr( & Bnew[JJ*L1+II-1] ) + l1 * get_global_cr( & Bnew[JJ*L1+II] );
  	
  	l1 = ( MG.Y_Bzold[iy] - get_global_cr( &(MG.Y_BzNew[JJ-1]) ) ) 
  			/ ( get_global_cr( &(MG.Y_BzNew[JJ]) ) - get_global_cr( &(MG.Y_BzNew[JJ-1]) ) );
  			
  	realw *row = (realw *)((char *)Bold + iy * BPitch);		
  	row[ix] = ((1.0 - l1) * a0 + l1 * a1)/(L1*M1);
	}
}

__global__ void Blinear_ByR2IR( realw *Bold, size_t BPitch, realw *Bnew,
															 const ModelPara *d_mp, ModelGrid MG, GridConv GC)
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int L0 = d_mp->L;
	const int M0 = d_mp->M;
	if(ix < L0 && iy < M0+1)
	{
		const int L1 = d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx;
		const int M1 = 2*d_mp->Any_New + d_mp->Ny;
  	
  	const unsigned int II = min( max( GC.xBzold2New[ix], 1 ), L1-1 );
  	const unsigned int JJ = min( max( GC.yoldBy2NewBz[iy], 1 ), M1-1 );
  	
  	realw Y, l1, a0, a1;
  	if(iy == M0)
  		Y = MG.Y_Bzold[M0-1] + 0.5 * MG.dy[M0-1];
  	else
  		Y = MG.Y_Bzold[iy] - 0.5 * MG.dy[iy];	
  		
  	l1 = ( MG.X_Bzold[ix] - get_global_cr( &(MG.X_BzNew[II-1]) ) ) 
  			/ ( get_global_cr( &(MG.X_BzNew[II]) ) - get_global_cr( &(MG.X_BzNew[II-1]) ) );
  			
  	a0 = (1.0 - l1) * get_global_cr( & Bnew[(JJ-1)*L1+II-1] ) + l1 * get_global_cr( & Bnew[(JJ-1)*L1+II] );	
  	a1 = (1.0 - l1) * get_global_cr( & Bnew[JJ*L1+II-1] ) + l1 * get_global_cr( & Bnew[JJ*L1+II] );			
  	
  	l1 = ( Y - get_global_cr( &(MG.Y_BzNew[JJ-1]) ) ) 
  			 / ( get_global_cr( &(MG.Y_BzNew[JJ]) ) - get_global_cr( &(MG.Y_BzNew[JJ-1]) ) );
  	
  	realw *row = (realw *)((char *)Bold + iy * BPitch);
  	row[ix] = ((1.0 - l1) * a0 + l1 * a1)/(L1*M1);
	}
}  	
  			
  	
__global__ void FBz_to_FBxFBy( WcuComplex *FBzC, WcuComplex *FBxC, WcuComplex *FByC,
															 const ModelPara *d_mp)
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	// NOTE L, and M in this file is the L1 and M1 elsewhere, this is stupid just to keep consistant with the cpu code
	const int L = d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx;
  const int M = 2*d_mp->Any_New + d_mp->Ny;
  if(ix < L/2+1 && iy < M)
  {
  	realw v1,u1,upcc;
  	WcuComplex a;
  	if(iy<M/2+1)
		{
			v1=(1.0*PI*iy)/M*(d_mp->dzmin/d_mp->dymin);
		}	
	  else
		{
			v1=-(1.0*PI*(M-iy))/M*(d_mp->dzmin/d_mp->dymin);
		}

		u1=(1.0*PI*ix)/L*(d_mp->dzmin/d_mp->dxmin);
	  
	  if(ix==0 && iy==0)
	  {
	  	// the (M*L) comes from the normalization of fft
	  	//a=iHilbert(FBzC[iy*(L/2+1)+ix])/(M*L);
			a=iHilbert(FBzC[iy*(L/2+1)+ix]);
	  	FBxC[iy*(L/2+1)+ix]=a;
	  	FByC[iy*(L/2+1)+ix]=a;
	  }
	  else
	  {
			// the (M*L) comes from the normalization of fft 
			//a=iHilbert(FBzC[iy*(L/2+1)+ix])/(M*L);
			a=iHilbert(FBzC[iy*(L/2+1)+ix]);

			// FIXME upcc is the main source of numerical error for the GPU computation
#ifdef USE_SINGLE_PRECISION
	  	//upcc = expf(-d_mp->dzmin/2.0f*sqrtf(u*u+v*v))*rsqrtf(u*u+v*v));
			upcc = expf(-sqrtf(u1*u1 + v1*v1)) *rsqrtf (u1*u1 + v1*v1) ;
#else
	  	//upcc = exp(-d_mp->dzmin/2.0*sqrt(u*u+v*v))*rsqrt(u*u+v*v);
			upcc = exp(-sqrt(u1*u1 + v1*v1)) * rsqrt (u1*u1 + v1*v1);
#endif	 

			FBxC[iy*(L/2+1)+ix]=(d_mp->dxmin/d_mp->dymin)*u1*upcc*a;
			FByC[iy*(L/2+1)+ix]=(d_mp->dymin/d_mp->dxmin)*v1*upcc*a;
			
	  } 
  }
}

int UpContinuation_gpu(DArrays *DPtrs, UpContP *UPP, ModelPara *MP, ModelPara *d_mp, 
											 ModelGrid *MG, ModelGrid *d_MG, GridConv *GC, GridConv *d_GC, 
											 cudaStream_t *stream)
{
  int L,M,L0,M0;
	
	M0=MP->M;
	L0=MP->L;
	// NOTE L, and M in this file is the L1 and M1 elsewhere
  L=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M=2*MP->Any_New+MP->Ny;

	cudaEvent_t Event;
	print_CUDA_error_if_any( cudaEventCreateWithFlags( &Event, cudaEventDisableTiming ), 510031 );
/*	
//--------------------------------------------------------------------------------------------------  
													// test, do spline interpolation on CPU 
	double wtime2,Runtime;																									
	//cpy Bz[0] to host device
	TRACE("cpy top slice of Bz to host to do upward continuation");
	print_CUDA_error_if_any( cudaMemcpy( UPP->h_Bz0[0], DPtrs->Bz, L0*M0*sizeof(realw), cudaMemcpyDeviceToHost), 41001);
	
  ///interpolation from the non-uniform mesh into regular mesh
  wtime2 = omp_get_wtime();
  SplineGridConvIR2R_cpu(MP,MG,GC,UPP->h_Bz0,UPP->h_BzAir);
  Runtime = omp_get_wtime()-wtime2;  
  printf("Spline Interpolation IR2R runtime=%8.5f sec.\n",Runtime);

	print_CUDA_error_if_any( cudaMemcpy( UPP->BzAir, UPP->h_BzAir[0], L*M*sizeof(realw), cudaMemcpyHostToDevice), 41002);
//-------------------------------------------------------------------------------------------------------  
*/
#if DEBUG == 2
	cudaEvent_t start, stop;
	start_timing_cuda(&start,&stop);	
#endif	

	SplineGridConvIR2R( MP, d_mp, d_MG, d_GC, UPP, DPtrs->Bz, stream );

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "Sprine interpolation IR2R");
	start_timing_cuda(&start,&stop);
#endif		
  
  // FFT real to complex plan1, run on stream[0]
#ifdef USE_SINGLE_PRECISION	
	print_cuFFT_error_if_any( cufftExecR2C(UPP->plan1, UPP->BzAir, UPP->FBzC), 41003);   
#else
	print_cuFFT_error_if_any( cufftExecD2Z(UPP->plan1, UPP->BzAir, UPP->FBzC), 41004); 
#endif  

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "fft");
	start_timing_cuda(&start,&stop);
#endif	

	// calculate FBxC and FByC on GPU
	dim3 threads(BLOCKSIZE2x,BLOCKSIZE2y2);
	dim3 grid( (L/2+1 + BLOCKSIZE2x-1)/BLOCKSIZE2x , (M + BLOCKSIZE2y2-1)/BLOCKSIZE2y2 );
	
	TRACE("Calculate FBx and FBy from FBz");
	FBz_to_FBxFBy<<<grid,threads, 0, stream[0] >>>( UPP->FBzC, UPP->FBxC, UPP->FByC, d_mp);
	GPU_ERROR_CHECKING("Calculate FBx and FBy from FBz");

	// ANCHOR Event
	print_CUDA_error_if_any( cudaEventRecord(Event, stream[0]), 510032 );

	// ANCHOR synchronisation
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[1], Event, 0), 510033 );


#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "Calculate FBx and FBy");
	start_timing_cuda(&start,&stop);
#endif		

  // ifft complex to real plan2 and plan3, 
  // TODO combine two iffts using cufftPlanMany 
#ifdef USE_SINGLE_PRECISION	
	print_cuFFT_error_if_any( cufftExecC2R(UPP->plan2, UPP->FBxC, UPP->FBxR), 41005);   
	print_cuFFT_error_if_any( cufftExecC2R(UPP->plan3, UPP->FByC, UPP->FByR), 41006); 
#else
	print_cuFFT_error_if_any( cufftExecZ2D(UPP->plan2, UPP->FBxC, UPP->FBxR), 41007); 
	print_cuFFT_error_if_any( cufftExecZ2D(UPP->plan3, UPP->FByC, UPP->FByR), 41008);
#endif    
	
#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "iffts");
	start_timing_cuda(&start,&stop);
#endif

	// linear interpolation from regular grid to iregular grid 
	dim3 grid1( (L0 + 1 + BLOCKSIZE2x-1)/BLOCKSIZE2x , (M0 + BLOCKSIZE2y2-1)/BLOCKSIZE2y2 );
	TRACE("Bilinear interpolation R2IR: BxAir");
	Blinear_BxR2IR<<<grid1,threads, 0, stream[0] >>>( DPtrs->BxAir, DPtrs->BxPitch, UPP->FBxR, d_mp, *d_MG, *d_GC);
	GPU_ERROR_CHECKING("Bilinear interpolation R2IR: BxAir");
	
	dim3 grid2( (L0 + BLOCKSIZE2x-1)/BLOCKSIZE2x , (M0 + 1 + BLOCKSIZE2y2-1)/BLOCKSIZE2y2 );
	TRACE("Bilinear interpolation R2IR: ByAir");
	Blinear_ByR2IR<<<grid2,threads, 0, stream[1] >>>( DPtrs->ByAir, DPtrs->ByPitch, UPP->FByR, d_mp, *d_MG, *d_GC);
	GPU_ERROR_CHECKING("Bilinear interpolation R2IR: ByAir");

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "Bilinear interpolation R2IR");
#endif	
	
	print_CUDA_error_if_any( cudaEventDestroy( Event ), 510034 );

  return 0;
}
