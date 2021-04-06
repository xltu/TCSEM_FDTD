/* load the initial value for the electronic filed and caculate the
initial value for the magnetic filed*/

#include<stdio.h>
#include<stdlib.h>
#include "FTDT.h"
extern double minsita;
extern double miu;
int LoadInitialValue(double ***Con,Efield *EF,Bfield *BF,ModelGrid *MG,ModelPara *MP,GridConv *GC,
                     UpContP *UPP,double T0,double dt1)
{
  FILE *Fp;
  int L,M,N,AnLx,AnRx,Any,Anz,Nx,Ny,Nz;
  double ***Ex1,***Ey1,***Ez1,sita,term1,term2;
  int i,j,k;


  AnLx=MP->AnLx_old;
  AnRx=MP->AnRx_old;
  Any=MP->Any_old;
  Anz=MP->Anz;
  Nx=MP->Nx;
  Ny=MP->Ny;
  Nz=MP->Nz;
  L=AnLx+AnRx+Nx;
  M=2*(Any)+Ny;
  N=Nz+Anz;

  ///the auxiliary memory for caculate the initial impulse value

	Ex1=Create_3D_Array(N+1,M+1,L);
	Ey1=Create_3D_Array(N+1,M,L+1);
	Ez1=Create_3D_Array(N,M+1,L+1);

  ///caculat the initial value of E impulse response
  for(k=0;k<=10;k++)
  {
    for(j=0;j<=(M-1)/2;j++)
    {
      for(i=0;i<=(L-1)/2;i++)
      {
        EF->Ex[k][j][i]=AnalyticalSolution(0,T0,i,j,k,minsita,MP,MG);
      }

      for(i=(L-1)/2+1;i<L;i++)
      {
        EF->Ex[k][j][i]=EF->Ex[k][j][(L-1)-i];
      }

    }

    /// symmetrical in the y direction
    for(j=(M-1)/2+1;j<=M;j++)
    {
      for(i=0;i<L;i++)
      {
        EF->Ex[k][j][i]=-EF->Ex[k][M-j][i];
      }
    }

    printf("Ex the %d layers\n",k);
  }

  for(k=0;k<=10;k++)
  {
    for(j=0;j<=(M-1)/2;j++)
    {
      for(i=0;i<=(L-1)/2;i++)
      {
        EF->Ey[k][j][i]=AnalyticalSolution(1,T0,i,j,k,minsita,MP,MG);
      }
      for(i=(L-1)/2+1;i<=L;i++)
      {
        EF->Ey[k][j][i]=-EF->Ey[k][j][L-i];
      }

    }

    for(j=(M-1)/2+1;j<M;j++)
    {
      for(i=0;i<=L;i++)
      {
        EF->Ey[k][j][i]=EF->Ey[k][M-1-j][i];
      }
    }

    printf("Ey the %d layers\n",k);
  }
//
//  for(k=0;k<=10;k++)
//  {
//    for(j=0;j<M+1;j++)
//    {
//      for(i=0;i<L+1;i++)
//      {
//        EF->Ez[k][j][i]=AnalyticalSolution(6,T0,i,j,k,minsita,MP,MG);
//      }
//    }
//    printf("Ez the %d layers\n",k);
//  }

/*
  for(k=0;k<=10;k++)
  {
    for(j=0;j<M;j++)
    {
      for(i=0;i<L+1;i++)
      {
        BF->Bx[k][j][i]=AnalyticalSolution(7,T0+dt1/2,i,j,k,minsita,MP,MG);
      }
    }
    printf("Bx the %d layers\n",k);
  }


  for(k=0;k<=10;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->By[k][j][i]=AnalyticalSolution(8,T0+dt1/2,i,j,k,minsita,MP,MG);
      }
    }
    printf("By the %d layers\n",k);
  }


  for(k=0;k<=10;k++)
  {
    for(j=0;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->Bz[k][j][i]=AnalyticalSolution(9,T0+dt1/2,i,j,k,minsita,MP,MG);
      }
    }
    printf("Bz the %d layers\n",k);
  }
*/
  ///the caculate of Bx/dt,change from step response to impulse response
  for(k=0;k<N;k++)
  {
    for(j=0;j<M;j++)
    {
      for(i=1;i<L;i++)
      {
        BF->Bx[k][j][i]=-(EF->Ez[k][j+1][i]-EF->Ez[k][j][i])/MG->dy[j]+
                        (EF->Ey[k+1][j][i]-EF->Ey[k][j][i])/MG->dz[k];
      }
    }
  }

  ///the caculate of By/dt,change from step response to impulse response
  for(k=0;k<N;k++)
  {
    for(j=1;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->By[k][j][i]=-(EF->Ex[k+1][j][i]-EF->Ex[k][j][i])/MG->dz[k]+
                        (EF->Ez[k][j][i+1]-EF->Ez[k][j][i])/MG->dx[i];
      }
    }
  }

  ///the caculate of Bz/dt,change from step response to impulse response
  for(j=0;j<M;j++)
  {
    for(i=0;i<L;i++)
      BF->Bz[N][j][i]=0.;
  }
  for(k=N-1;k>=0;k--)
  {
    for(j=0;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->Bz[k][j][i]=BF->Bz[k+1][j][i]+MG->dz[k]/MG->dx[i]*(BF->Bx[k][j][i+1]-BF->Bx[k][j][i])+
                                        MG->dz[k]/MG->dy[j]*(BF->By[k][j+1][i]-BF->By[k][j][i]);
        //BF->Bz[k][j][i]=-(EF->Ey[k][j][i+1]-EF->Ey[k][j][i])/MG->dx[i]+
         //               (EF->Ex[k][j+1][i]-EF->Ex[k][j][i])/MG->dy[j];
      }
    }
  }

  UpContinuation(UPP,MP,MG,GC);

  ///caculate the dE/dt, change the E step response into impulse response
  /// dEx/dt
  for(k=0;k<N+1;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L;i++)
      {
        if(k==0)
        {
          if(j==0)
          {
            sita=0.5*Con[k][j][i];
            term1=(BF->Bz[0][0][i])/(2*MG->dy[0]);
            term2=-(BF->By[0][0][i]-UPP->ByAir[0][i])/(2*MG->dz[0]);
            EF->Ex[k][j][i]=0;
          }

          else if(j==M)
          {
            sita=0.5*Con[k][j-1][i];
            term1=(-BF->Bz[0][M-1][i])/(2*MG->dy[M-1]);
            term2=-(BF->By[0][M][i]-UPP->ByAir[M][i])/(2*MG->dz[0]);
            EF->Ex[k][j][i]=0;
          }

          else
          {
            sita=0.25*(Con[k][j][i]+Con[k][j-1][i]);
            term1=(BF->Bz[0][j][i]-BF->Bz[0][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            term2=-(BF->By[0][j][i]-UPP->ByAir[j][i])/(2*MG->dz[0]);
            EF->Ex[k][j][i]=2*(term1+term2)/(sita*miu);
          }

        }
        else if(k==N)
        {
          if(j==0)
          {
            sita=Con[k-1][j][i];
            term1=(BF->Bz[N][j][i])/(2*MG->dy[j]);
            term2=-(-BF->By[N-1][j][i])/(2*MG->dz[N-1]);
            EF->Ex[k][j][i]=0;
          }
          else if(j==M)
          {
            sita=Con[k-1][j-1][i];
            term1=(-BF->Bz[N][j-1][i])/(2*MG->dy[j-1]);
            term2=-(-BF->By[N-1][j][i])/(2*MG->dz[N-1]);
            EF->Ex[k][j][i]=0;
          }
          else
          {
            sita=0.5*(Con[k-1][j][i]+Con[k-1][j-1][i]);
            term1=(BF->Bz[N][j][i]-BF->Bz[N][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            term2=-(-BF->By[N-1][j][i])/(2*MG->dz[N-1]);
            EF->Ex[k][j][i]=0;
          }
        }
        else
        {
          if(j==0)
          {
            sita=0.5*(Con[k][j][i]+Con[k-1][j][i]);
            term1=(BF->Bz[k][0][i])/(2*MG->dy[0]);
            term2=-(BF->By[k][0][i]-BF->By[k-1][0][i])/(MG->dz[k]+MG->dz[k-1]);
            EF->Ex[k][j][i]=0;
          }
          else if(j==M)
          {
            sita=0.5*(Con[k][j-1][i]+Con[k-1][j-1][i]);
            term1=(-BF->Bz[k][M-1][i])/(2*MG->dy[M-1]);
            term2=-(BF->By[k][M][i]-BF->By[k-1][M][i])/(MG->dz[k]+MG->dz[k-1]);
            EF->Ex[k][j][i]=0;
          }
          else
          {
            sita=0.25*(Con[k][j][i]+Con[k][j-1][i]+Con[k-1][j][i]+Con[k-1][j-1][i]);
            term1=(BF->Bz[k][j][i]-BF->Bz[k][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            term2=-(BF->By[k][j][i]-BF->By[k-1][j][i])/(MG->dz[k]+MG->dz[k-1]);
            EF->Ex[k][j][i]=2*(term1+term2)/(sita*miu);
          }
        }

        //EF->Ex[k][j][i]=2*(term1+term2)/(sita*miu);
      }
    }
  }

  ///dEy/dt
  for(k=0;k<N+1;k++)
  {
    for(j=0;j<M;j++)
    {
      for(i=0;i<L+1;i++)
      {
        if(k==0)
        {
          if(i==0)
          {
            sita=0.5*Con[k][j][i];
            term1=(BF->Bx[0][j][0]-UPP->BxAir[j][0])/(2*MG->dz[0]);
            term2=-(BF->Bz[0][j][0])/(2*MG->dx[0]);
            EF->Ey[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=0.5*Con[k][j][i-1];
            term1=(BF->Bx[0][j][L]-UPP->BxAir[j][L])/(2*MG->dz[0]);
            term2=-(-BF->Bz[0][j][L-1])/(2*MG->dx[L-1]);
            EF->Ey[k][j][i]=0;
          }
          else
          {
            sita=0.25*(Con[k][j][i]+Con[k][j][i-1]);
            term1=(BF->Bx[0][j][i]-UPP->BxAir[j][i])/(2*MG->dz[0]);
            term2=-(BF->Bz[0][j][i]-BF->Bz[0][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            EF->Ey[k][j][i]=2*(term1+term2)/(sita*miu);
          }
        }
        else if(k==N)
        {
          if(i==0)
          {
            sita=Con[k-1][j][i];
            term1=(-BF->Bx[N-1][j][i])/(2*MG->dz[N-1]);
            term2=-(BF->Bz[N][j][i])/(2*MG->dx[i]);
            EF->Ey[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=Con[k-1][j][i-1];
            term1=(-BF->Bx[N-1][j][i])/(2*MG->dz[N-1]);
            term2=-(-BF->Bz[N][j][i-1])/(2*MG->dx[i-1]);
            EF->Ey[k][j][i]=0;
          }
          else
          {
            sita=0.5*(Con[k-1][j][i]+Con[k-1][j][i-1]);
            term1=(-BF->Bx[N-1][j][i])/(2*MG->dz[N-1]);
            term2=-(BF->Bz[N][j][i]-BF->Bz[N][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            EF->Ey[k][j][i]=0;
          }
        }
        else
        {
          if(i==0)
          {
            sita=0.5*(Con[k][j][i]+Con[k-1][j][i]);
            term1=(BF->Bx[k][j][0]-BF->Bx[k-1][j][0])/(MG->dz[k]+MG->dz[k-1]);
            term2=-(BF->Bz[k][j][0])/(2*MG->dx[0]);
            EF->Ey[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=0.5*(Con[k][j][i-1]+Con[k-1][j][i-1]);
            term1=(BF->Bx[k][j][L]-BF->Bx[k-1][j][L])/(MG->dz[k]+MG->dz[k-1]);
            term2=-(-BF->Bz[k][j][L-1])/(2*MG->dx[L-1]);
            EF->Ey[k][j][i]=0;
          }
          else
          {
            sita=0.25*(Con[k][j][i]+Con[k][j][i-1]+Con[k-1][j][i]+Con[k-1][j][i-1]);
            term1=(BF->Bx[k][j][i]-BF->Bx[k-1][j][i])/(MG->dz[k]+MG->dz[k-1]);
            term2=-(BF->Bz[k][j][i]-BF->Bz[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            EF->Ey[k][j][i]=2*(term1+term2)/(sita*miu);
          }
        }

        //EF->Ey[k][j][i]=2*(term1+term2)/(sita*miu);
      }
    }
  }

  ///dEz/dt
  for(k=0;k<N;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L+1;i++)
      {
        if(j==0)
        {
          if(i==0)
          {
            sita=Con[k][j][i];
            term1=(BF->By[k][j][i])/(2*MG->dx[i]);
            term2=-(BF->Bx[k][j][i])/(2*MG->dy[j]);
            EF->Ez[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=Con[k][j][i-1];
            term1=(-BF->By[k][j][i-1])/(2*MG->dx[i-1]);
            term2=-(BF->Bx[k][j][i])/(2*MG->dy[j]);
            EF->Ez[k][j][i]=0;
          }
          else
          {
            sita=0.5*(Con[k][j][i]+Con[k][j][i-1]);
            term1=(BF->By[k][j][i]-BF->By[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            term2=-(BF->Bx[k][j][i])/(2*MG->dy[j]);
            EF->Ez[k][j][i]=0;
          }
        }
        else if(j==M)
        {
          if(i==0)
          {
            sita=Con[k][j-1][i];
            term1=(BF->By[k][j][i])/(2*MG->dx[i]);
            term2=-(-BF->Bx[k][j-1][i])/(2*MG->dy[j-1]);
            EF->Ez[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=Con[k][j-1][i-1];
            term1=(-BF->By[k][j][i-1])/(2*MG->dx[i-1]);
            term2=-(-BF->Bx[k][j-1][i])/(2*MG->dy[j-1]);
            EF->Ez[k][j][i]=0;
          }
          else
          {
            sita=0.5*(Con[k][j-1][i]+Con[k][j-1][i-1]);
            term1=(BF->By[k][j][i]-BF->By[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            term2=-(-BF->Bx[k][j-1][i])/(2*MG->dy[j-1]);
            EF->Ez[k][j][i]=0;
          }
        }
        else
        {
          if(i==0)
          {
            sita=0.5*(Con[k][j][i]+Con[k][j-1][i]);
            term1=(BF->By[k][j][i])/(2*MG->dx[i]);
            term2=-(BF->Bx[k][j][i]-BF->Bx[k][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            EF->Ez[k][j][i]=0;
          }
          else if(i==L)
          {
            sita=0.5*(Con[k][j][i-1]+Con[k][j-1][i-1]);
            term1=(-BF->By[k][j][i-1])/(2*MG->dx[i-1]);
            term2=-(BF->Bx[k][j][i]-BF->Bx[k][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            EF->Ez[k][j][i]=0;
          }
          else
          {
            sita=0.25*(Con[k][j][i]+Con[k][j][i-1]+Con[k][j-1][i]+Con[k][j-1][i-1]);
            term1=(BF->By[k][j][i]-BF->By[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
            term2=-(BF->Bx[k][j][i]-BF->Bx[k][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
            EF->Ez[k][j][i]=2*(term1+term2)/(miu*sita);
          }
        }
        //EF->Ez[k][j][i]=2*(term1+term2)/(miu*sita);
      }
    }
  }

  ///caculat the initial value of E impulse response
  for(k=0;k<=10;k++)
  {
    for(j=0;j<=(M-1)/2;j++)
    {
      for(i=0;i<=(L-1)/2;i++)
      {
        Ex1[k][j][i]=AnalyticalSolution(0,T0+dt1/2,i,j,k,minsita,MP,MG);
      }

      for(i=(L-1)/2+1;i<L;i++)
      {
        Ex1[k][j][i]=Ex1[k][j][(L-1)-i];
      }

    }

    for(j=(M-1)/2+1;j<=M;j++)
    {
      for(i=0;i<L;i++)
      {
        Ex1[k][j][i]=-Ex1[k][M-j][i];
      }
    }

    printf("Ex the %d layers\n",k);
  }

  for(k=0;k<=10;k++)
  {
    for(j=0;j<=(M-1)/2;j++)
    {
      for(i=0;i<=(L-1)/2;i++)
      {
        Ey1[k][j][i]=AnalyticalSolution(1,T0+dt1/2,i,j,k,minsita,MP,MG);
      }

      for(i=(L-1)/2+1;i<=L;i++)
      {
        Ey1[k][j][i]=-Ey1[k][j][L-i];
      }

    }

    for(j=(M-1)/2+1;j<M;j++)
    {
      for(i=0;i<=L;i++)
      {
        Ey1[k][j][i]=Ey1[k][M-1-j][i];
      }
    }

    printf("Ey the %d layers\n",k);
  }


  ///the caculate of Bx/dt,change from step response to impulse response
  for(k=0;k<N;k++)
  {
    for(j=0;j<M;j++)
    {
      for(i=1;i<L;i++)
      {
        BF->Bx[k][j][i]=-(Ez1[k][j+1][i]-Ez1[k][j][i])/MG->dy[j]+
                        (Ey1[k+1][j][i]-Ey1[k][j][i])/MG->dz[k];
      }
    }
  }

  ///the caculate of By/dt,change from step response to impulse response
  for(k=0;k<N;k++)
  {
    for(j=1;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->By[k][j][i]=-(Ex1[k+1][j][i]-Ex1[k][j][i])/MG->dz[k]+
                        (Ez1[k][j][i+1]-Ez1[k][j][i])/MG->dx[i];
      }
    }
  }

  ///the caculate of Bz/dt,change from step response to impulse response
  for(j=0;j<M;j++)
  {
    for(i=0;i<L;i++)
      BF->Bz[N][j][i]=0.;
  }
  for(k=N-1;k>=0;k--)
  {
    for(j=0;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        BF->Bz[k][j][i]=BF->Bz[k+1][j][i]+MG->dz[k]/MG->dx[i]*(BF->Bx[k][j][i+1]-BF->Bx[k][j][i])+
                                      MG->dz[k]/MG->dy[j]*(BF->By[k][j+1][i]-BF->By[k][j][i]);
         //BF->Bz[k][j][i]=-(Ey1[k][j][i+1]-Ey1[k][j][i])/MG->dx[i]+
          //              (Ex1[k][j+1][i]-Ex1[k][j][i])/MG->dy[j];
      }
    }
  }

  Free_3D_Array(Ex1,N+1);
  Free_3D_Array(Ey1,N+1);
  Free_3D_Array(Ez1,N);

  if((Fp=fopen("AnalyticEx.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**EF->Ex,sizeof(double),(N+1)*(M+1)*L,Fp);
  fclose(Fp);

  if((Fp=fopen("AnalyticEy.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**EF->Ey,sizeof(double),(N+1)*M*(L+1),Fp);
  fclose(Fp);

  if((Fp=fopen("AnalyticEz.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**EF->Ez,sizeof(double),N*(M+1)*(L+1),Fp);
  fclose(Fp);



  if((Fp=fopen("AnalyticBx.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**BF->Bx,sizeof(double),N*M*(L+1),Fp);
  fclose(Fp);

  if((Fp=fopen("AnalyticBy.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**BF->By,sizeof(double),N*(M+1)*L,Fp);
  fclose(Fp);

  if((Fp=fopen("AnalyticBz.dat","wb"))==NULL)
  {
    printf("fail to open the AnalyticalEz data file!\n");
    getchar();
    exit(1);
  }
  fwrite(**BF->Bz,sizeof(double),(N+1)*M*L,Fp);
  fclose(Fp);

  return 0;
}

