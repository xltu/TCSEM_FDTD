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
 
/* set up the FDTD modeling and
 load in the parameters */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"FTDT.h"
#include<omp.h>
int ModelingSetup(double ****Con,ModelPara *MP,ModelGrid *MG,double *sfa1,double *sfa2,double *sita_t,int *N_Src_Steps,double *minsita)
{
	TRACE("Loading modeling mesh, conductivity model, and source waveform paramters\n");
	FILE *FsetIn,*FcondIn;
	int L,M,N;
	int n1,nc,i,j,k;	
	double a0,logcond,x0,y0,z0;
	
	//---------------------------------------load in setup file-------------------------------------------------------------------------
	if((FsetIn=fopen("FDTD_setup.in","r"))==NULL)
	{
		printf("Fail to open the FDTD_setup.in file!\n");
		perror("Error: ");
		getchar();
		return(1);
	}
	
	fscanf(FsetIn,"%lf %lf\n", sfa1, sfa2);	// time step speeding up factor and numda stabilizing factor
	fscanf(FsetIn,"%d %lf\n", &n1, sita_t);	 
	*N_Src_Steps=n1/2;
	fscanf(FsetIn,"%lf %lf %lf\n", &x0, &y0, &z0);
	fscanf(FsetIn,"%lf %lf %lf\n", &(MP->dxmin), &(MP->dymin), &(MP->dzmin));
	fscanf(FsetIn,"%lf %lf\n", &(MP->arg1), &(MP->arg2));
	fscanf(FsetIn,"%d %d %d %d %d\n", &(MP->ALx2), &(MP->ALx1), &(MP->Nx), &(MP->ARx1), &(MP->ARx2));
	fscanf(FsetIn,"%d %d %d\n", &(MP->Ay2), &(MP->Ay1), &(MP->Ny));
	fscanf(FsetIn,"%d %d %d\n", &(MP->Az2), &(MP->Az1), &(MP->Nz));
	
	fclose(FsetIn);
	
// support version 1.0 model parameter	
#ifdef MP_VER_1	
	MP->AnLx_old=(MP->ALx2)+(MP->ALx1);
	MP->AnRx_old=(MP->ARx2)+(MP->ARx1);
	
	MP->Any_old=(MP->Ay2)+(MP->Ay1);
	MP->Anz=(MP->Az2)+(MP->Az1);
#endif

#if MAXDEBUG == 1
	printf("Grids in x dir: %d %d %d %d %d\n", (MP->ALx2), (MP->ALx1), (MP->Nx), (MP->ARx1), (MP->ARx2));
	printf("Grids in y dir: %d %d %d %d %d\n", (MP->Ay2),  (MP->Ay1),  (MP->Ny), (MP->Ay1), (MP->Ay2));
	printf("Grids in z dir: %d %d %d\n", (MP->Nz), (MP->Az1), (MP->Az2));
#endif	
	//---------------------------------------------------------------------------------------------------
	//					Creating the modeling grid
	L=(MP->ALx2)+(MP->ALx1)+(MP->ARx2)+(MP->ARx1)+MP->Nx;
	M=2*(MP->Ay2 + MP->Ay1)+MP->Ny;
	N=MP->Nz+(MP->Az2)+(MP->Az1);
	MP->L=L;
	MP->M=M;
	MP->N=N;
	
	MG->dx=(double *)malloc(L*sizeof(double));
	MG->dy=(double *)malloc(M*sizeof(double));
	MG->dz=(double *)malloc(N*sizeof(double));
	MG->X_Bzold=(double *)malloc(L*sizeof(double));
	MG->Y_Bzold=(double *)malloc(M*sizeof(double));
	MG->Z=(double *)malloc(N*sizeof(double));
  	
	if(MG->dx==NULL||MG->dy==NULL||MG->dz==NULL||MG->X_Bzold==NULL||MG->Y_Bzold==NULL)
	{
		printf("fail to get memory for ModelGrid\n");
		getchar();
		exit(EXIT_FAILURE);
	}
	
	// seting up cell sizes
	for(i=(MP->ALx2)+(MP->ALx1);i<MP->Nx+(MP->ALx2)+(MP->ALx1);i++)
		MG->dx[i]=MP->dxmin;
	for(i=(MP->ALx2)+(MP->ALx1)-1;i>=(MP->ALx2);i--)
	{
		MG->dx[i]=MP->dxmin * pow( MP->arg1 , double(MP->ALx1 + MP->ALx2-i) ); 
#if DEBUG == 1
		if( MG->dx[i]!= MG->dx[i+1] * MP->arg1  && abs(MG->dx[i] - MG->dx[i+1] * MP->arg1) > 1e-3*MG->dx[i] )
			 printf("Error in the x direction mesh size %d\n",i);
#endif		
	}	
	for(i=(MP->ALx2)-1;i>=0;i--)
	{
		MG->dx[i]= MP->dxmin * pow( MP->arg1 , double(MP->ALx1) )*pow( MP->arg2 , double(MP->ALx2-i) );
#if DEBUG == 1
		if ( MG->dx[i] != MG->dx[i+1]*(MP->arg2) && abs(MG->dx[i] - MG->dx[i+1]*(MP->arg2) ) > 1e-3*MG->dx[i] )
			printf("Error in the x direction mesh size %d\n",i);
#endif			
	}	
	for(i=MP->Nx+(MP->ALx2)+(MP->ALx1);i<MP->Nx+(MP->ALx2)+(MP->ALx1)+(MP->ARx1);i++)
	{
		MG->dx[i] = MP->dxmin * pow( MP->arg1 , double(i- MP->ALx1 - MP->ALx2- MP->Nx + 1) ) ;
#if DEBUG == 1
		if ( MG->dx[i] !=  MG->dx[i-1]*(MP->arg1) && abs(MG->dx[i] - MG->dx[i-1]*(MP->arg1) ) > 1e-3*MG->dx[i] )
			printf("Error in the x direction mesh size %d\n",i);
#endif			
	}
	for(i=MP->Nx+(MP->ALx2)+(MP->ALx1)+(MP->ARx1);i<L;i++)
	{
		MG->dx[i] = MP->dxmin * pow( MP->arg1 , double(MP->ARx1) ) * pow( MP->arg2 , double(i -MP->Nx -(MP->ALx2)-(MP->ALx1)-(MP->ARx1) + 1) );
#if DEBUG ==1		
		if ( MG->dx[i] !=  MG->dx[i-1]*(MP->arg2) && abs(MG->dx[i] - MG->dx[i-1]*(MP->arg2)) > 1e-3*MG->dx[i] )
			printf("Error in the x direction mesh size %d\n",i);
#endif			
	}	

	for(j=(MP->Ay2)+(MP->Ay1);j<MP->Ny+(MP->Ay2)+(MP->Ay1);j++)
		MG->dy[j]=MP->dymin;
	for(j=(MP->Ay2)+(MP->Ay1)-1;j>=(MP->Ay2);j--)
		MG->dy[j]=MG->dy[j+1]*(MP->arg1);
	for(j=(MP->Ay2)-1;j>=0;j--)
		MG->dy[j]=MG->dy[j+1]*(MP->arg2);
	for(j=MP->Ny+(MP->Ay2)+(MP->Ay1);j<MP->Ny+(MP->Ay2)+2*(MP->Ay1);j++)
		MG->dy[j]=MG->dy[j-1]*(MP->arg1);
	for(j=MP->Ny+(MP->Ay2)+2*(MP->Ay1);j<M;j++)
		MG->dy[j]=MG->dy[j-1]*(MP->arg2);

	for(k=0;k<MP->Nz;k++)
		MG->dz[k]=MP->dzmin;
	for(k=MP->Nz;k<MP->Nz+(MP->Az1);k++)
		MG->dz[k]=MG->dz[k-1]*(MP->arg1);
	for(k=MP->Nz+(MP->Az1);k<N;k++)
		MG->dz[k]=MG->dz[k-1]*(MP->arg2);	
	
	///caculate the old coordinates
	if( (MP->Nx)%2 == 0 )	// even number of center grids
	{
		nc=MP->Nx/2+(MP->ALx2)+(MP->ALx1);
		MG->X_Bzold[nc]=0.5*MG->dx[nc]+x0;
	}
	else
	{
		nc=(MP->Nx-1)/2+(MP->ALx2)+(MP->ALx1);
		MG->X_Bzold[nc]=x0;
	}
	for(i=nc+1;i<L;i++)
		MG->X_Bzold[i]=MG->X_Bzold[i-1]+0.5*MG->dx[i-1]+0.5*MG->dx[i];
	for(i=nc-1;i>=0;i--)
		MG->X_Bzold[i]=MG->X_Bzold[i+1]-0.5*MG->dx[i+1]-0.5*MG->dx[i];
	
	if( (MP->Ny)%2 == 0 )	// even number of center grids
	{
		nc=MP->Ny/2+(MP->Ay2)+(MP->Ay1);
		MG->Y_Bzold[nc]=0.5*MG->dy[nc]+y0;
	}
	else
	{
		nc=(MP->Ny-1)/2+(MP->Ay2)+(MP->Ay1);
		MG->Y_Bzold[nc]=y0;
	}
	for(j=nc+1;j<M;j++)
		MG->Y_Bzold[j]=MG->Y_Bzold[j-1]+0.5*MG->dy[j-1]+0.5*MG->dy[j];	
	for(j=nc-1;j>=0;j--)
		MG->Y_Bzold[j]=MG->Y_Bzold[j+1]-0.5*MG->dy[j+1]-0.5*MG->dy[j];
		
	MG->Z[0]=0.5*MG->dz[0]+z0;
	for(i=1;i<N;i++)
		MG->Z[i]=MG->Z[i-1]+0.5*MG->dz[i-1]+0.5*MG->dz[i];	
		
	// the grid size mapping into a regular grid	
	MP->AnLx_New=ceil( (MG->X_Bzold[(MP->ALx2)+(MP->ALx1)-1] - MG->X_Bzold[0] + MG->dx[(MP->ALx2)+(MP->ALx1)-1]/2 + MG->dx[0]/2)/MP->dxmin );
	MP->AnRx_New=ceil( (MG->X_Bzold[L-1] - MG->X_Bzold[L-(MP->ARx1)-(MP->ARx2)] + MG->dx[L-1]/2 + MG->dx[L-(MP->ARx1)-(MP->ARx2)]/2)/MP->dxmin );
	MP->Any_New=ceil( (MG->Y_Bzold[(MP->Ay2)+(MP->Ay1)-1] - MG->Y_Bzold[0] + MG->dy[(MP->Ay2)+(MP->Ay1)-1]/2 + MG->dy[0]/2)/MP->dymin );	
		
	MP->LNx2= pow(2,floor (log(MP->AnLx_New+MP->AnRx_New+MP->Nx)/log(2.)) + 1);
	MP->LMy2= pow(2,floor (log(2*MP->Any_New+MP->Ny)/log(2.)) + 1);
	
	TRACE("Modeling mesh was set successfully\n");
	
	//-----------------------------------------------------------------------------------------------
	///					load in conductivity model
	(*Con)=Create_3D_Array(N,M,L);
	if( (FcondIn=fopen("Cond.bin","rb")) ==NULL )
	{
		printf("fail to open the conductivity model file: Cond.bin\n");
		getchar();
		exit(1);
	}
	fread(***Con,sizeof(double),N*M*L,FcondIn);
	fclose(FcondIn);

	// check the background conductivity TODO faster check on GPU??
	logcond=0;
	n1=0;
	#pragma omp parallel for collapse(3) private(k,j,i,a0) reduction(+: logcond,n1)
	for(k=0;k<N;k++)
		for(j=0;j<M;j++)
			for(i=0;i<L;i++)
			{
				a0=log10( (*Con)[k][j][i] );
				if(  a0 > 1e-2 && a0 <= 0 )
				{
					n1++;
					logcond+=a0;
				}
			}
				
	if( n1 > 0 )
		*minsita = pow10(logcond/n1);
	else
		*minsita=0.1;
	
	TRACE("Conductivity model was loaded successfully\n");		
  return 0;
}
