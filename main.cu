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
 /*REVIEW Corrections and Updates
 *@Xiaolei Tu @Feb.26,2021, cocurrent kernel execution using multiple streams (4)    
 *@Xiaolei Tu @Mar.25,2021, optimize the stream event functions to enable performance boost   
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<unistd.h>
#include<omp.h>
#include<fftw3.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include "FTDT.h"
#include "Cuda_device.h"
#include "prepare_constants_cuda.h"

/*
The variables readed constantly all threads or most of the threads: 
	miu, nmuda, wave, and dt, Current? we should set it as constant memory
	dx,dy,dz   float(?)(double is to expensive ) type constant memory?
	
	cond, EMF are double type global memory (maybe cond could be set as float?)
*/

double miu=4*PI*1.e-7;
int MAXTHREAD=0;

int main()
{  
  int NTx,NRx;	// number of transmitter positions and receiver positions
  RxPos xyzRx;	// Rx positions SOA, for convenience of data transfer between host and device
  TxPos *xyzTx; //Tx positions AOS
  RxTime tRx;	// recording time channels
  	
  /// the sturctures
  ModelPara MP;
  ModelGrid MG;
  UpContP UPP;
  GridConv GC;
  
  int L,M,L1,M1,itx;
  double ***Con0;
	
  double nmuda,mindx,minsita=1.0/10;
  double toT,Tstart=0,Tend=10,dt1=0,dt2;
  int n=0,Nt=0,ne_efs=0,nb_efs=0;
  double ***efs;
  double Tpof[2][3]={0};	
  
  FILE *Fout;
  char FOname[LFILE];
  double sfa1,sfa2;  

  double sita_t=5.0e-4;			// sita_t is related to the bandwidth of impulse function
  int N_Src_Steps=100;
  double *Wave,*dt;
  
  double wtime1,wtime2,wtime3,Runtime;
  wtime1 = omp_get_wtime();
  MAXTHREAD=omp_get_max_threads();	
	
  //innitizlization of the conductivity model
	if( ModelingSetup(&Con0, &MP, &MG, &sfa1, &sfa2, &sita_t, &N_Src_Steps,&minsita) )
	{
		printf("Error: unable to loading the setup parameters or model parameters\n");
		getchar();
		exit(EXIT_FAILURE);
	}
	else
		printf("The model is initialized successfully!\n");
		
	L1=MP.AnLx_New+MP.AnRx_New+MP.Nx;
  M1=2*MP.Any_New+MP.Ny;

  L=MP.L;
  M=MP.M;
	mindx=min(MP.dzmin,min(MP.dxmin,MP.dymin));
  ///the caculation of time steps
  Wave=(double *)malloc((2*N_Src_Steps+1)*sizeof(double));
  dt=(double *)malloc((2*N_Src_Steps+2)*sizeof(double));

  if(Wave==NULL||dt==NULL)
  {
    printf("fail to get memory for sourcewave!\n");
    getchar();
    exit(EXIT_FAILURE);
  }
  
  Waveform(N_Src_Steps,sita_t,dt,Wave);

  MG.X_BzNew=(double *)malloc(L1*sizeof(double));
  MG.Y_BzNew=(double *)malloc(M1*sizeof(double));

  GC.xBzNew2old=(int *)malloc(L1*sizeof(int));
  GC.yBzNew2old=(int *)malloc(M1*sizeof(int));
  GC.xBzold2New=(int *)malloc(L*sizeof(int));
  GC.yBzold2New=(int *)malloc(M*sizeof(int));
  GC.xoldBx2NewBz=(int *)malloc((L+1)*sizeof(int));
  GC.yoldBy2NewBz=(int *)malloc((M+1)*sizeof(int));

  if(MG.X_BzNew==NULL||MG.Y_BzNew==NULL||GC.xBzNew2old==NULL||
     GC.yBzNew2old==NULL||GC.xBzold2New==NULL||GC.yBzold2New==NULL||
     GC.xoldBx2NewBz==NULL||GC.yoldBy2NewBz==NULL)
  {
    printf("fail to get memory for GridConversion...ect in the main function\n");
    getchar();
    exit(1);
  }
	
	TRACE("all the memory on host are allocated successfully!\n");
	
  // setting up grids		
  ModelGridMapping(&MP,&MG,&GC);
	
  // loading the Rx and Tx parameters, must after grid setting up
  LoadRxTxPos(MP, MG, &tRx, &xyzRx, &xyzTx, &NRx, &NTx);
  
  TRACE("Tx and Tx positions are loaded successfully!\n");
  //----------------------------- precalculate the total number of time iterations-----------------------------------------------
  ///the total number of time steps
  Tend=tRx.Tn;
  Tstart=6*sita_t;
  toT=Tstart;
  Nt=2*N_Src_Steps+1;
  dt1=min(0.15*sfa1*sqrt(miu*(toT)*minsita/8)*mindx,1.2*dt[2*N_Src_Steps]);

  while(toT <= Tend)
  {
	dt2=min(0.15*sfa1*sqrt(miu*(toT)*minsita/8)*mindx,1.2*dt1);
	Nt++;
	toT+=dt2;
    dt1=dt2;
  }
  Nt++;
  printf("time iteration =%d\n",Nt);
  
  //------------------------------------------------------setup GPU device memory----------------------------------------
  int N_device, myrank=0;
  initialize_cuda_device( myrank , &N_device);
  wtime3 = omp_get_wtime();
  // 3D dimensional pitched mem for EM fields and Cond
	DArrays DPtrs;
	// Rx positions
	RxPos d_xyzRx;
	// Rx list
	int *RxLst;
	ModelGrid d_MG;
	GridConv d_GC;
	prepare_device_arrays( &DPtrs, MP, Con0, &d_xyzRx, xyzRx, NRx, &RxLst, &d_MG, MG, &d_GC, GC);
	//print_CUDA_error_if_any( cudaMalloc((void **)&RxLst, NRx*sizeof(int)), 1025);
	
	//cpy model parameter structure to device
	ModelPara *d_mp=setConst_MP(&MP);
	
	realw *d_efs, *d_ef0;

  //Cuda streams and events
  cudaStream_t stream[4];
  for(int ist=0; ist<4; ist++)
    print_CUDA_error_if_any( cudaStreamCreate( &stream[ist] ), 1501 );

  //-------------------------------------------setup up ward continuation----------------------------------------------------
	initialize_UpCont_gpu_resources(&UPP, &MP, d_mp, &d_MG, stream);
  //--------------------------------------------------loop over each tx------------------------------------------------------
for(itx=0; itx<NTx; itx++)
{  
  printf("start to model the EM field generated by Tx %d\n",itx+1);
  //cpy the Tx structure to device
  TxPos *d_Tx=setConst_Tx(&(xyzTx[itx]));
  
  //cpy Rx list to device
  print_CUDA_error_if_any(cudaMemcpy((int *)RxLst, xyzTx[itx].RxNum, xyzTx[itx].nr*sizeof(int),cudaMemcpyHostToDevice),1050);
  
  // set the cuda memory to zeros
  set_zeros_EM_arrays( &DPtrs, &MP );
  
  print_CUDA_error_if_any( cudaMalloc((void **)&d_efs, xyzTx[itx].nr * tRx.NComp * tRx.Nt * sizeof(realw)), 1051);
  print_CUDA_error_if_any( cudaMalloc((void **)&d_ef0, xyzTx[itx].nr * tRx.NComp * 2 * sizeof(realw)), 1052);
  
	efs=Create_3D_Array(xyzTx[itx].nr,tRx.NComp,tRx.Nt);

	toT=0;
	dt1=dt[n];
	dt2=dt[n+1];
	
	Tpof[0][1]=toT+dt1;
	Tpof[1][1]=toT+dt1+0.5*dt2;
	
	while( ne_efs<tRx.Nt ||  ( nb_efs<tRx.Nt && tRx.nefcmp<tRx.NComp ) )
	{
		
    //printf("the %dth of %d time iteration, total time=%10.8fs\n",n,Nt,toT+dt1/2);
    nmuda=7.5*sfa2*dt2*dt1/miu/mindx/mindx;
#ifdef FFTW    
    UpContinuation(&DPtrs, &UPP, &MP, &MG, &GC);
#else
		UpContinuation_gpu(&DPtrs, &UPP, &MP, d_mp, &MG, &d_MG, &GC, &d_GC, stream);
#endif    

    // E field time=toT+dt1 after the following iteration;
#if DEBUG == 1    
    wtime2 = omp_get_wtime();
#endif    
    if(n<2*N_Src_Steps)
      //ElectricalField(Con,&EF,&BF,&UPP,&MP,&MG,nmuda,dt1,Wave[n],xyzTx[itx]);	
      ElectricalField( &DPtrs, &MP, d_mp, &MG,	nmuda, dt1,	Wave[n], xyzTx[itx], d_Tx, stream);
    else
      //ElectricalField(Con,&EF,&BF,&UPP,&MP,&MG,nmuda,dt1,0,xyzTx[itx]);
      ElectricalField( &DPtrs, &MP, d_mp, &MG,	nmuda, dt1,	0, xyzTx[itx], d_Tx, stream);
#if DEBUG == 1       
    Runtime = omp_get_wtime()-wtime2;  
    printf("E field updating runtime=%8.5f sec.\n",Runtime);
     
    wtime2 = omp_get_wtime();
#endif    
		// B field time=toT+dt1+dt2/2. afeter the following itration
    //Bderivate(&EF,&BF,&MP,&MG,dt1,dt2,toT);
    Bderivate(&DPtrs, MP, d_mp, &d_MG, 0.5*(dt1+dt2), stream);
#if DEBUG == 1    
    Runtime = omp_get_wtime()-wtime2;  
    printf("B field updating runtime=%8.5f sec.\n",Runtime);
#endif        
		toT+=dt1;
		dt1=dt2;
		n++;
		
		if(n<2*N_Src_Steps)
			dt2=dt[n+1];
		else
		 	dt2=min(0.15*sfa1*sqrt(miu*toT*minsita/8.)*mindx,1.2*dt1);
		
		// interpolate the EM field to the receiver position
		TRACE("Interpolate to Rx\n");
#if DEBUG == 1 		
		wtime2 = omp_get_wtime();
#endif		
		//EFInterp2Rx(&ne_efs, &nb_efs, tRx, xyzTx[itx], xyzRx, MG, &EF,&BF, toT+dt1, dt2, Tpof, efs, ef0);
		EFInterp2Rx(&ne_efs, &nb_efs, tRx, xyzTx[itx], &d_xyzRx, RxLst, &d_MG, d_mp, &DPtrs, toT+dt1, dt2, Tpof, d_efs, d_ef0, stream);
#if DEBUG == 1								
		Runtime = omp_get_wtime()-wtime2;  
    printf("Interpolate to Rx runtime=%8.5f sec.\n",Runtime);	 
#endif   	
	}

	printf("end of time iteration for Tx %d\n", itx+1);
	
	//cpy data from device to host
	print_CUDA_error_if_any(cudaMemcpy(efs[0][0], d_efs, xyzTx[itx].nr * tRx.NComp * tRx.Nt * sizeof(realw),cudaMemcpyDeviceToHost),1053);
	
	snprintf(FOname,LFILE, "Tx%d.bout",itx+1);
	if((Fout=fopen(FOname,"wb"))==NULL)
	{
		printf("fail to open the output Txxx.out file!\n");
		getchar();
		exit(1);
	}
	fwrite(&(xyzTx[itx].nr),sizeof(int),1,Fout);
	fwrite(&(tRx.NComp),sizeof(int),1,Fout);
	fwrite(&(tRx.Nt),sizeof(int),1,Fout);
  fwrite(efs[0][0],sizeof(double),(xyzTx[itx].nr)*(tRx.NComp)*(tRx.Nt),Fout);

	fclose(Fout);
	
	//Free_3D_Array(ef0,2);
	Free_3D_Array(efs,xyzTx[itx].nr);
	print_CUDA_error_if_any( cudaFree( d_ef0 ), 1054);
	print_CUDA_error_if_any( cudaFree( d_efs ), 1055);
}	

	release_UpCont_gpu_resources(&UPP);

  for(int ist=0; ist<4; ist++)
    print_CUDA_error_if_any( cudaStreamDestroy( stream[ist] ), 1503 );

	//Free_3D_Array(Con0,N);
	
	FreeRxTxArray(&tRx, &xyzRx, &xyzTx, NRx, NTx);
	
	free(MG.dx);free(MG.dy);free(MG.dz);
	free(MG.X_BzNew);free(MG.X_Bzold);free(MG.Y_BzNew);free(MG.Y_Bzold);free(MG.Z);
	free(GC.xBzNew2old);free(GC.yBzNew2old);free(GC.xBzold2New);free(GC.yBzold2New);
	free(GC.xoldBx2NewBz);free(GC.yoldBy2NewBz);
	
	//----------------------------Realise device memory-------------------------------------
	Device_cleanup( &DPtrs , &d_xyzRx , &RxLst , &d_MG, &d_GC ); 
	
	Runtime = omp_get_wtime()-wtime1;
	printf("FDTD modleing completed, runtime=%8.1f sec.\n",Runtime);

  Runtime = omp_get_wtime()-wtime3;
  printf("Pure GPU runtime=%8.1f sec.\n",Runtime);

	return 0;
}
