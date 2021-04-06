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
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<unistd.h>
#include "FTDT.h"

#ifdef FFTW
#include<omp.h>
#include<fftw3.h>
#else
#include<cufft.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "Cuda_device.h"
#endif

void initialize_fft(UpContP *UPP, ModelPara *MP, cudaStream_t *stream)
{
	TRACE("Initialize fft/ifft plans\n");
	int M,L,M1,L1;
	
	L1=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M1=2*MP->Any_New+MP->Ny;

  L=MP->L;
  M=MP->M;
#ifdef FFTW
	char FfftwWP[LFILE];
	
	UPP->Bz0	=	Create2DArray(M,L);
  UPP->BxAir=Create2DArray(M,L+1);
  UPP->ByAir=Create2DArray(M+1,L);

  UPP->BzAir=Create2DArray(M1,L1);
  UPP->FBxR=Create2DArray(M1,L1);
  UPP->FByR=Create2DArray(M1,L1);

  UPP->FBzC=Create2DfftwArray(M1,(L1)/2+1);
  UPP->FBxC=Create2DfftwArray(M1,(L1)/2+1);
  UPP->FByC=Create2DfftwArray(M1,(L1)/2+1);
  
  fftw_init_threads();
  fftw_plan_with_nthreads(MAXTHREAD);	
  
  if( snprintf(FfftwWP,LFILE, "FDTDWisdomFFTWplanThread%d.dat",MAXTHREAD) > LFILE )
  {
  	printf("Error: the wisdom fftw plan data file name is too long\n");
  	exit(-1);
  }
  if(~access(FfftwWP,R_OK)) 
  	fftw_import_wisdom_from_filename(FfftwWP);
  UPP->plan1=fftw_plan_dft_r2c_2d(M1,L1,&(UPP->BzAir[0][0]),&(UPP->FBzC[0][0]),FFTW_PATIENT|FFTW_DESTROY_INPUT);
  UPP->plan2=fftw_plan_dft_c2r_2d(M1,L1,&(UPP->FBxC[0][0]),&(UPP->FBxR[0][0]),FFTW_PATIENT|FFTW_DESTROY_INPUT);
  UPP->plan3=fftw_plan_dft_c2r_2d(M1,L1,&(UPP->FByC[0][0]),&(UPP->FByR[0][0]),FFTW_PATIENT|FFTW_DESTROY_INPUT);
  if( access(FfftwWP,W_OK) )
  {
  	printf("Warning: have no write permission to the wisdom fftw plan dir, save the new one to local dir\n");
  	snprintf(FfftwWP,LFILE, "FDTDWisdomFFTWplanThread%d.dat",MAXTHREAD);
  	fftw_export_wisdom_to_filename(FfftwWP);
  }
  else	
  	fftw_export_wisdom_to_filename(FfftwWP);
#else
	
	// TODO for test purpose only, shall be removed in a future version
	UPP->h_Bz0	=	Create2DArray(M,L);
	UPP->h_BzAir=Create2DArray(M1,L1);
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->FBxR), 	M1*L1*sizeof(realw)), 4001);
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->FByR), 	M1*L1*sizeof(realw)), 4002);	
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->BzAir), M1*L1*sizeof(realw)), 4003);
	
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->FBzC), M1*(L1/2+1)*sizeof(WcuComplex)), 4004);
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->FBxC), M1*(L1/2+1)*sizeof(WcuComplex)), 4005);
	print_CUDA_error_if_any( cudaMalloc((void **)&(UPP->FByC), M1*(L1/2+1)*sizeof(WcuComplex)), 4006);
#ifdef USE_SINGLE_PRECISION	
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan1), M1, L1, CUFFT_R2C), 4007);
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan2), M1, L1, CUFFT_C2R), 4008);
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan3), M1, L1, CUFFT_C2R), 4009);
#else
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan1), M1, L1, CUFFT_D2Z), 4011);
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan2), M1, L1, CUFFT_Z2D), 4012);
	print_cuFFT_error_if_any( cufftPlan2d(&(UPP->plan3), M1, L1, CUFFT_Z2D), 4013);
#endif	

	print_cuFFT_error_if_any( cufftSetStream(UPP->plan1, stream[0]), 400001);
	print_cuFFT_error_if_any( cufftSetStream(UPP->plan2, stream[0]), 400002);
	print_cuFFT_error_if_any( cufftSetStream(UPP->plan3, stream[1]), 400003);
	
	GPU_ERROR_CHECKING("Initialize fft/ifft plans!");
		
#endif
}

void destroy_fft(UpContP *UPP)
{
	TRACE("Destroy fft/ifft data and plans\n");
#ifdef FFTW
	fftw_cleanup_threads();
	fftw_destroy_plan(UPP->plan1);
	fftw_destroy_plan(UPP->plan2);
	fftw_destroy_plan(UPP->plan3);
	Free2DfftwArray(UPP->FBzC);
	Free2DfftwArray(UPP->FBxC);
	Free2DfftwArray(UPP->FByC);
	Free2DArray(UPP->Bz0);
	Free2DArray(UPP->BzAir);
	Free2DArray(UPP->FBxR);
	Free2DArray(UPP->FByR);
	Free2DArray(UPP->BxAir);
	Free2DArray(UPP->ByAir);
#else
	print_cuFFT_error_if_any( cufftDestroy(UPP->plan1), 4014);
	print_cuFFT_error_if_any( cufftDestroy(UPP->plan2), 4015);
	print_cuFFT_error_if_any( cufftDestroy(UPP->plan3), 4016);
	
	//TODO XXX for test only, should be removed
	Free2DArray(UPP->h_Bz0);
	Free2DArray(UPP->h_BzAir);
	
	print_CUDA_error_if_any( cudaFree( (UPP->FBxR) ), 4017);
	print_CUDA_error_if_any( cudaFree( (UPP->FByR) ), 4018);
	print_CUDA_error_if_any( cudaFree( (UPP->BzAir) ), 4019);
	
	print_CUDA_error_if_any( cudaFree( (UPP->FBxC) ), 4020);
	print_CUDA_error_if_any( cudaFree( (UPP->FByC) ), 4020);
	print_CUDA_error_if_any( cudaFree( (UPP->FBzC) ), 4020);
	GPU_ERROR_CHECKING("Destroy fft/ifft data and plans\n");
#endif	
}

