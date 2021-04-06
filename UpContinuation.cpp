/* the UpContinuation function for the DB */
#include<stdio.h>
#include<stdlib.h>
#include<fftw3.h>
#include<math.h>
#include"FTDT.h"
#include "omp.h"
int UpContinuation(DArrays *DPtrs, UpContP *UPP, ModelPara *MP, ModelGrid *MG, GridConv *GC)
{
  int i,j;
  double v,u,upcc;
  int L,M,L2,M2,DL,DM;
	double wtime2,Runtime;
	
  L=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M=2*MP->Any_New+MP->Ny;
  
	//XXX cpy Bz[0] to host device
	CP2Host_bz0_UPP(DPtrs, UPP->Bz0[0], MP->M, MP->L);
	
 // L2=MP->LNx2;
 // M2=MP->LMy2;
 // DL=(int)((L2-L)/2);
 // DM=(int)((M2-M)/2);
    DL=0;DM=0;
  ///interpolation from the non-uniform mesh into regular mesh
  wtime2 = omp_get_wtime();
  SplineGridConvIR2R(MP,MG,GC,UPP->Bz0,UPP->BzAir);
  Runtime = omp_get_wtime()-wtime2;  
  printf("Spline Interpolation IR2R runtime=%8.5f sec.\n",Runtime);
  /*for(j=DM;j<M+DM;j++)
    for(i=DL;i<L+DL;i++)
      UPP->BzAir[j][i]=UPP->Bz0[j-DM][i-DL];
  */
 /* for(j=0;j<DM;j++)
    for(i=DL;i<L+DL;i++)
      UPP->BzAir[j][i]=0.5*(1-cos(PI*j/DM))*UPP->BzAir[DM][i];
  for(j=M+DM;j<M2;j++)
    for(i=DL;i<L+DL;i++)
      UPP->BzAir[j][i]=0.5*(1-cos(PI*(M2-j)/(M2-M-DM+1)))*UPP->BzAir[M+DM-1][i];

  for(j=0;j<M2;j++)
  {
    for(i=0;i<DL;i++)
      UPP->BzAir[j][i]=0.5*(1-cos(PI*i/DL))*UPP->BzAir[j][DL];
    for(i=L+DL;i<L2;i++)
      UPP->BzAir[j][i]=0.5*(1-cos(PI*(L2-i)/(L2-L-DL+1)))*UPP->BzAir[j][L+DL-1];
  }
*/
/*  ///=====================test code below==============================
  FILE *fp;
  if((fp=fopen("testUPContinuation.dat","wb"))==NULL)
  {
    printf("fail to create the test file");
    getchar();
    exit(EXIT_FAILURE);
  }
  fwrite(&M,sizeof(int),1,fp);
  fwrite(&L,sizeof(int),1,fp);
  fwrite(&M,sizeof(int),1,fp);
  fwrite(&L,sizeof(int),1,fp);
  fwrite(UPP->BzAir[0],sizeof(double),M*L,fp);
  ///=======================test code above===============================
*/
  wtime2 = omp_get_wtime();	
  fftw_execute(UPP->plan1);
  Runtime = omp_get_wtime()-wtime2;  
  printf("FFT runtime=%8.5f sec.\n",Runtime);
  	
  #pragma omp parallel for collapse(2) private(j,i,v,u,upcc)	
  for(j=0;j<M;j++)
  {
    for(i=0;i<L/2+1;i++)
    {
      if(j<M/2+1)
	    v=2*PI*j/M/MP->dymin;
	  else
	    v=-2*PI*(M-j)/(M)/MP->dymin;
		  
      u=2*PI*i/L/MP->dxmin;
      if(i==0 && j==0)
      {
        UPP->FBxC[j][i][0]=-UPP->FBzC[j][i][1];
        UPP->FBxC[j][i][1]=UPP->FBzC[j][i][0];
        UPP->FByC[j][i][0]=-UPP->FBzC[j][i][1];
        UPP->FByC[j][i][1]=UPP->FBzC[j][i][0];
      }
      else
      {
        upcc=exp(-MP->dzmin/2*sqrt(u*u+v*v))/sqrt(u*u+v*v);
        UPP->FBxC[j][i][0]=-u*upcc*UPP->FBzC[j][i][1];
        UPP->FBxC[j][i][1]=u*upcc*UPP->FBzC[j][i][0];
        UPP->FByC[j][i][0]=-v*upcc*UPP->FBzC[j][i][1];
        UPP->FByC[j][i][1]=v*upcc*UPP->FBzC[j][i][0];
      }
    }
  }
	

/*  for(j=0;j<M;j++)
  {
    if(j<M/2+1)
      v=2*PI*j/M/MP->dymin;
    else
      v=-2*PI*(M-j)/(M)/MP->dymin;
    for(i=0;i<L/2+1;i++)
    {
      u=2*PI*i/L/MP->dxmin;
      if(i==0 && j==0)
      {
        UPP->FBxC[j][i][0]=-UPP->FBzC[j][i][1];
        UPP->FBxC[j][i][1]=UPP->FBzC[j][i][0];
        UPP->FByC[j][i][0]=-UPP->FBzC[j][i][1];
        UPP->FByC[j][i][1]=UPP->FBzC[j][i][0];
      }
      else
      {
        upcc=exp(-MP->dzmin/2*sqrt(u*u+v*v))/sqrt(u*u+v*v);
        UPP->FBxC[j][i][0]=u*upcc*(sin(MP->dzmin/2*u)*UPP->FBzC[j][i][0]-cos(MP->dzmin/2*u)*UPP->FBzC[j][i][1]);
        UPP->FBxC[j][i][1]=u*upcc*(sin(MP->dzmin/2*u)*UPP->FBzC[j][i][1]+cos(MP->dzmin/2*u)*UPP->FBzC[j][i][0]);
        UPP->FByC[j][i][0]=v*upcc*(sin(MP->dzmin/2*v)*UPP->FBzC[j][i][0]-cos(MP->dzmin/2*v)*UPP->FBzC[j][i][1]);
        UPP->FByC[j][i][1]=v*upcc*(sin(MP->dzmin/2*v)*UPP->FBzC[j][i][1]+cos(MP->dzmin/2*v)*UPP->FBzC[j][i][0]);
      }
    }
  }
*/
 /* ///==============test code below==================================
  fwrite(UPP->FBzC[0],sizeof(fftw_complex),M*(L/2+1),fp);
  fwrite(UPP->FBxC[0],sizeof(fftw_complex),M*(L/2+1),fp);
  fwrite(UPP->FByC[0],sizeof(fftw_complex),M*(L/2+1),fp);
  ///===============test code above==================================
*/
  wtime2 = omp_get_wtime();		
  fftw_execute(UPP->plan2);
  fftw_execute(UPP->plan3);
  Runtime = omp_get_wtime()-wtime2;  
  printf("IFFT runtime=%8.5f sec.\n",Runtime);
  
  #pragma omp parallel for private(i,j)
  for(j=0;j<M;j++)
  {
    for(i=0;i<L;i++)
    {
      UPP->FBxR[j][i]/=(M*L);
      UPP->FByR[j][i]/=(M*L);
    }
  }
  
  wtime2 = omp_get_wtime();	
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			BilinearGridConvR2IR(0,MP,MG,GC,UPP->FBxR,UPP->BxAir);
		}
		
		#pragma omp section
		{
			BilinearGridConvR2IR(1,MP,MG,GC,UPP->FByR,UPP->ByAir);
		}
	}
	
	Runtime = omp_get_wtime()-wtime2;  
  printf("Bilinear interpolation R2IR runtime=%8.5f sec.\n",Runtime);

/*  for(j=0;j<M;j++)
    for(i=1;i<L;i++)
      UPP->BxAir[j][i]=UPP->FBxR[j+DM][i+DL];

  for(j=1;j<M;j++)
    for(i=0;i<L;i++)
      UPP->ByAir[j][i]=UPP->FByR[j+DM][i+DL];
*/
 /* ///================test code below======================================
  fwrite(UPP->FBxR[0],sizeof(double),M*L,fp);
  fwrite(UPP->FByR[0],sizeof(double),M*L,fp);
  fwrite(UPP->BxAir[0],sizeof(double),M*(L+1),fp);
  fwrite(UPP->ByAir[0],sizeof(double),(M+1)*L,fp);
  fclose(fp);
  ///==================test code above=====================================
*/
 /* for(j=0;j<M;j++)
  {
    UPP->BxAir[j][0]=1./((M)*(L))*(1.5*UPP->FBxR[j][0]-0.5*UPP->FBxR[j][1]);
    UPP->BxAir[j][L]=1./((M)*(L))*(1.5*UPP->FBxR[j][L-1]-0.5*UPP->FBxR[j][L-2]);
    for(i=1;i<L;i++)
    {
      UPP->BxAir[j][i]=1./((M)*(L))*(UPP->FBxR[j][i]+UPP->FBxR[j][i-1])/2;
      UPP->BxAir[j][i]=  UPP->FBxR[j][i];
      UPP->ByAir[j][i]=  UPP->FByR[j][i];
    }
  }

  /// interpolation from the regular mesh into the defined ununiform mesh
  for(j=0;j<M+1;j++)
  {
    for(i=0;i<L;i++)
    {
      if(j==0)
        UPP->ByAir[j][i]=1./((M)*(L))*(1.5*UPP->FByR[0][i]-0.5*UPP->FByR[1][i]);
      else if(j==M)
        UPP->ByAir[j][i]=1./((M)*(L))*(1.5*UPP->FByR[M-1][i]-0.5*UPP->FByR[M-2][i]);
      else
        UPP->ByAir[j][i]=1./((M)*(L))*(UPP->FByR[j][i]+UPP->FByR[j-1][i])/2;
    }
  }
*/
	//XXX copy from host to device
	CP2Device_bxby_UPP(DPtrs, UPP->BxAir[0], UPP->ByAir[0], MP->M, MP->L);
  return 0;
}
