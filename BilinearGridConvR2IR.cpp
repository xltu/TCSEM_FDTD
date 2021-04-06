#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include <omp.h>
#include "FTDT.h"
int BilinearGridConvR2IR(int tag,ModelPara *MP,ModelGrid *MG,GridConv *GC,double **Bnew,double **Bold)

/*int BilinearGridInvR2IR(int tag,double **Bnew,double **Bold,double *MG->dx,double *MG->dy,
                      int L0,int L1,int M0,int M1,
                      double *MG->X_Bzold,double *MG->Y_Bzold,double *MG->X_BzNew,double *MG->Y_BzNew,
                      int *GC->xBzold2New,int *GC->yBzold2New,int *GC->xoldBx2NewBz,int *GC->yoldBy2NewBz)*/
{
  double a1,a2,hx1,hx2,hy1,hy2,k1,k2;
  int L0,L1,M0,M1;
  int i,j,II,JJ;
  double *X,*Y;
  
  L0=MP->AnLx_old+MP->AnRx_old+MP->Nx;
  L1=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M0=2*MP->Any_old+MP->Ny;
  M1=2*MP->Any_New+MP->Ny;

  if(tag==0)
  {
    ///caculate of Bx using linear interpolation


    ///the Bx coordinates in the old grid
    X=(double *)malloc((L0+1)*sizeof(double));
    memcpy(X,MG->X_Bzold,L0*sizeof(double));
    
    #pragma omp parallel for private(i)
    for(i=0;i<L0;i++)
      X[i]=X[i]-MG->dx[i]/2;
    X[L0]=X[L0-1]+MG->dx[L0-1];  
	
	#pragma omp parallel for collapse(2) private(j,i,II,JJ,hx1,hx2,k1,k2,a1,a2,hy1,hy2)
    for(j=0;j<M0;j++)
    {
      for(i=0;i<L0+1;i++)
      {
        II=GC->xoldBx2NewBz[i];
        JJ=GC->yBzold2New[j];
        if(II==0)
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ][II+1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ+1][II+1]-Bnew[JJ+1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ-2][II+1]-Bnew[JJ-2][II])/hx1;
            k2=(Bnew[JJ-1][II+1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ-2][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ][II+1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ-1][II+1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
        else if(II==L1)
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-X[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II-2])/hx1;
            k2=(Bnew[JJ+1][II-1]-Bnew[JJ+1][II-2])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-X[i];
            k1=(Bnew[JJ-2][II-1]-Bnew[JJ-2][II-2])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II-2])/hx1;
            a1=Bnew[JJ-2][II-2]-k1*hx2;
            a2=Bnew[JJ-1][II-2]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-X[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II-2])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II-2])/hx1;
            a1=Bnew[JJ][II-2]-k1*hx2;
            a2=Bnew[JJ-1][II-2]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
        else
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ+1][II-1]-Bnew[JJ+1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ-2][II-1]-Bnew[JJ-2][II])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ-2][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-X[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-MG->Y_Bzold[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
      }
    }

    free(X);
  }
  else if(tag==1)
  {
    ///caculate By
    Y=(double *)malloc((M0+1)*sizeof(double));
    memcpy(Y,MG->Y_Bzold,M0*sizeof(double));
    
    #pragma omp parallel for private(i)
    for(i=0;i<M0;i++)
      Y[i]=Y[i]-MG->dy[i]/2;
    Y[M0]=Y[M0-1]+MG->dy[M0-1];  
      
	#pragma omp parallel for collapse(2) private(j,i,II,JJ,hx1,hx2,k1,k2,a1,a2,hy1,hy2)
    for(j=0;j<M0+1;j++)
    {
      for(i=0;i<L0;i++)
      {
        II=GC->xBzold2New[i];
        JJ=GC->yoldBy2NewBz[j];
        if(II==0)
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II+1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ+1][II+1]-Bnew[JJ+1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ-2][II+1]-Bnew[JJ-2][II])/hx1;
            k2=(Bnew[JJ-1][II+1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ-2][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II+1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II+1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ-1][II+1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
        else if(II==L1)
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II-2])/hx1;
            k2=(Bnew[JJ+1][II-1]-Bnew[JJ+1][II-2])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-MG->X_Bzold[i];
            k1=(Bnew[JJ-2][II-1]-Bnew[JJ-2][II-2])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II-2])/hx1;
            a1=Bnew[JJ-2][II-2]-k1*hx2;
            a2=Bnew[JJ-1][II-2]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II-2];
            hx2=MG->X_BzNew[II-2]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II-2])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II-2])/hx1;
            a1=Bnew[JJ][II-2]-k1*hx2;
            a2=Bnew[JJ-1][II-2]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
        else
        {
          if(JJ==0)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ+1][II-1]-Bnew[JJ+1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ+1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ+1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else if(JJ==M1)
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ-2][II-1]-Bnew[JJ-2][II])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ-2][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ-2];
            hy2=MG->Y_BzNew[JJ-2]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
          else
          {
            hx1=MG->X_BzNew[II-1]-MG->X_BzNew[II];
            hx2=MG->X_BzNew[II]-MG->X_Bzold[i];
            k1=(Bnew[JJ][II-1]-Bnew[JJ][II])/hx1;
            k2=(Bnew[JJ-1][II-1]-Bnew[JJ-1][II])/hx1;
            a1=Bnew[JJ][II]-k1*hx2;
            a2=Bnew[JJ-1][II]-k2*hx2;

            hy1=MG->Y_BzNew[JJ-1]-MG->Y_BzNew[JJ];
            hy2=MG->Y_BzNew[JJ]-Y[j];
            k1=(a2-a1)/hy1;
            Bold[j][i]=a1-k1*hy2;
          }
        }
      }
    }
    free(Y);
  }
  else
  {
    printf("wrong tag for Bilinear interpolation\n");
    getchar();
    exit(EXIT_FAILURE);
  }

  return 0;

}
