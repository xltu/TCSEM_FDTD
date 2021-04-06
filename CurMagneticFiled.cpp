/*the renew of Cur*H */
#include<omp.h>
#include"FTDT.h"
extern double miu;
int CurMagneticField(double ***Curxh,double ***Curyh,double ***Curzh,double ***DBx,double ***DBy,double ***DBz,
                     double **AirBx,double **AirBy,ModelPara *MP,ModelGrid *MG,double dt1,double dt2)
/*int CurMagneticField(double ***Curxh,double ***Curyh,double ***Curzh,double ***DBx,double ***DBy,double ***DBz,
                     double **AirBx,double **AirBy,int L,int M,int N,double *dx,double *dy,double *dz,double dt1,double dt2)
*/
{
  int L,M,N,i,j,k;
  double term1,term2,term3,Coeff;

  L=MP->AnLx_old+MP->AnRx_old+MP->Nx;
  M=2*MP->Any_old+MP->Ny;
  N=MP->Anz+MP->Nz;

 /* /// Curxh
  for(i=0;i<L;i++)
  {
    Curxh[0][0][i]-=(dt1+dt2)/miu*(DBy[0][0][i]-AirBy[0][i])/(2*dz[0]);
    Curxh[0][M][i]-=(dt1+dt2)/miu*(DBy[0][M][i]-AirBy[M][i])/(2*dz[0]);
  }
  for(j=1;j<M;j++)
  {
    for(i=0;i<L;i++)
    {
      Curxh[0][j][i]+=(dt1+dt2)/miu*((DBz[0][j][i]-DBz[0][j-1][i])/(dy[j]+dy[j-1])-(DBy[0][j][i]-AirBy[j][i])/(2*dz[0]));
    }
  }
  for(k=1;k<N;k++)
  {
    for(i=0;i<L;i++)
    {
      Curxh[k][0][i]-=(dt1+dt2)/miu*(DBy[k][0][i]-DBy[k-1][0][i])/(dz[k]+dz[k-1]);
      Curxh[k][M][i]-=(dt1+dt2)/miu*(DBy[k][M][i]-DBy[k-1][M][i])/(dz[k]+dz[k-1]);
    }
    for(j=1;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        Curxh[k][j][i]+=(dt1+dt2)/miu*((DBz[k][j][i]-DBz[k][j-1][i])/(dy[j]+dy[j-1])-(DBy[k][j][i]-DBy[k-1][j][i])/(dz[k]+dz[k-1]));
      }
    }
  }
  for(j=1;j<M;j++)
  {
    for(i=0;i<L;i++)
    {
      Curxh[N][j][i]+=(dt1+dt2)/miu*(DBz[N][j][i]-DBz[N][j-1][i])/(dy[j]+dy[j-1]);
    }
  }

  ///curyh
  for(j=0;j<M;j++)
  {
    Curyh[0][j][0]+=(dt1+dt2)/miu*(DBx[0][j][0]-AirBx[j][0])/(2*dz[0]);
    Curyh[0][j][L]+=(dt1+dt2)/miu*(DBx[0][j][L]-AirBx[j][L])/(2*dz[0]);
  }
  for(j=0;j<M;j++)
  {
    for(i=1;i<L;i++)
    {
      Curyh[0][j][i]+=(dt1+dt2)/miu*((DBx[0][j][i]-AirBx[j][i])/(2*dz[0])-(DBz[0][j][i]-DBz[0][j][i-1])/(dx[i]+dx[i-1]));
    }
  }

  for(k=1;k<N;k++)
  {
    for(j=0;j<M;j++)
    {
      Curyh[k][j][0]+=(dt1+dt2)/miu*(DBx[k][j][0]-DBx[k-1][j][0])/(dz[k]+dz[k-1]);
      Curyh[k][j][L]+=(dt1+dt2)/miu*(DBx[k][j][L]-DBx[k-1][j][L])/(dz[k]+dz[k-1]);
    }
    for(j=0;j<M;j++)
    {
      for(i=1;i<L;i++)
      {
        Curyh[k][j][i]+=(dt1+dt2)/miu*((DBx[k][j][i]-DBx[k-1][j][i])/(dz[k]+dz[k-1])-(DBz[k][j][i]-DBz[k][j][i-1])/(dx[i]+dx[i-1]));
      }
    }
  }

  for(j=0;j<M;j++)
  {
    for(i=1;i<L;i++)
    {
      Curyh[N][j][i]-=(dt1+dt2)/miu*(DBz[N][j][i]-DBz[N][j][i-1])/(dx[i]+dx[i-1]);
    }
  }

  ///curzh
  for(k=0;k<N;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L+1;i++)
      {
        if(k==0)
        {
          term1=0;
          Coeff=2*dz[0];
        }
        else
        {
          term1=Curzh[k-1][j][i];
          Coeff=dz[k]+dz[k-1];
        }


        if(j==0)
          term2=0;
        else if(j==M)
          term2=0;
        else
          term2=(Curyh[k][j][i]-Curyh[k][j-1][i])/(dy[j]+dy[j-1]);

        if(i==0)
          term3=0;
        else if(i==L)
          term3=0;
        else
          term3=(Curxh[k][j][i]-Curxh[k][j][i-1])/(dx[i]+dx[i-1]);

        Curzh[k][j][i]=term1-Coeff*(term2+term3);

      }
    }
  } */

/*  ///curzh
  for(k=0;k<N;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L+1;i++)
      {

        if(j==0 || j==M)
          term3=0;
        else
          term3=(DBx[k][j][i]-DBx[k][j-1][i])/(dy[j]+dy[j-1]);

        if(i==0 || i==L)
          term2=0;
        else
          term2=(DBy[k][j][i]-DBy[k][j][i-1])/(dx[i]+dx[i-1]);

        Curzh[k][j][i]+=(dt1+dt2)/miu*(term2-term3);

      }
    }
  } */


    /// Curxh
  #pragma omp parallel for if (L>200) private(i)  
  for(i=0;i<L;i++)
  {
    //Curxh[0][0][i]-=(dt1+dt2)/miu*(DBy[0][0][i]-AirBy[0][i])/(2*MG->dz[0]);
    //Curxh[0][M][i]-=(dt1+dt2)/miu*(DBy[0][M][i]-AirBy[M][i])/(2*MG->dz[0]);
    Curxh[0][0][i]+=(dt1+dt2)/miu*((DBz[0][0][i])/(2*MG->dy[0])-(DBy[0][0][i]-AirBy[0][i])/(2*MG->dz[0]));
    Curxh[0][M][i]+=(dt1+dt2)/miu*((-DBz[0][M-1][i])/(2*MG->dy[M-1])-(DBy[0][M][i]-AirBy[M][i])/(2*MG->dz[0]));
  }
  #pragma omp parallel for collapse(2) private(j,i) 
  for(j=1;j<M;j++)
  {
    for(i=0;i<L;i++)
    {
      Curxh[0][j][i]+=(dt1+dt2)/miu*((DBz[0][j][i]-DBz[0][j-1][i])/(MG->dy[j]+MG->dy[j-1])-(DBy[0][j][i]-AirBy[j][i])/(2*MG->dz[0]));
    }
  }
  #pragma omp parallel for collapse(2) private(k,i) 
  for(k=1;k<N;k++)
  {
    for(i=0;i<L;i++)
    {
      //Curxh[k][0][i]-=(dt1+dt2)/miu*(DBy[k][0][i]-DBy[k-1][0][i])/(MG->dz[k]+MG->dz[k-1]);
      //Curxh[k][M][i]-=(dt1+dt2)/miu*(DBy[k][M][i]-DBy[k-1][M][i])/(MG->dz[k]+MG->dz[k-1]);
      Curxh[k][0][i]+=(dt1+dt2)/miu*((DBz[k][0][i])/(2*MG->dy[0])-(DBy[k][0][i]-DBy[k-1][0][i])/(MG->dz[k]+MG->dz[k-1]));
      Curxh[k][M][i]+=(dt1+dt2)/miu*((-DBz[k][M-1][i])/(2*MG->dy[M-1])-(DBy[k][M][i]-DBy[k-1][M][i])/(MG->dz[k]+MG->dz[k-1]));
    }
   }
   
  #pragma omp parallel for collapse(2) private(k,j,i) 
  for(k=1;k<N;k++)
  {    
    for(j=1;j<M;j++)
    {
      for(i=0;i<L;i++)
      {
        Curxh[k][j][i]+=(dt1+dt2)/miu*((DBz[k][j][i]-DBz[k][j-1][i])/(MG->dy[j]+MG->dy[j-1])-(DBy[k][j][i]-DBy[k-1][j][i])/(MG->dz[k]+MG->dz[k-1]));
      }
    }
  }
  
  #pragma omp parallel for collapse(2) private(j,i) 
  for(j=1;j<M;j++)
  {
    for(i=0;i<L;i++)
    {
      //Curxh[N][j][i]+=(dt1+dt2)/miu*(DBz[N][j][i]-DBz[N][j-1][i])/(MG->dy[j]+MG->dy[j-1]);
      Curxh[N][j][i]+=(dt1+dt2)/miu*((DBz[N][j][i]-DBz[N][j-1][i])/(MG->dy[j]+MG->dy[j-1])-(-DBy[N-1][j][i])/(2*MG->dz[N-1]));
    }
  }

  ///curyh
  #pragma omp parallel for if (M>200) private(j) 
  for(j=0;j<M;j++)
  {
    //Curyh[0][j][0]+=(dt1+dt2)/miu*(DBx[0][j][0]-AirBx[j][0])/(2*MG->dz[0]);
    //Curyh[0][j][L]+=(dt1+dt2)/miu*(DBx[0][j][L]-AirBx[j][L])/(2*MG->dz[0]);
    Curyh[0][j][0]+=(dt1+dt2)/miu*((DBx[0][j][0]-AirBx[j][0])/(2*MG->dz[0])-(DBz[0][j][0])/(2*MG->dx[0]));
    Curyh[0][j][L]+=(dt1+dt2)/miu*((DBx[0][j][L]-AirBx[j][L])/(2*MG->dz[0])-(-DBz[0][j][L-1])/(2*MG->dx[L-1]));
  }
  #pragma omp parallel for collapse(2) private(j,i)
  for(j=0;j<M;j++)
  {
    for(i=1;i<L;i++)
    {
      Curyh[0][j][i]+=(dt1+dt2)/miu*((DBx[0][j][i]-AirBx[j][i])/(2*MG->dz[0])-(DBz[0][j][i]-DBz[0][j][i-1])/(MG->dx[i]+MG->dx[i-1]));
    }
  }
  #pragma omp parallel for collapse(2) private(k,j)
  for(k=1;k<N;k++)
  {
    for(j=0;j<M;j++)
    {
      //Curyh[k][j][0]+=(dt1+dt2)/miu*(DBx[k][j][0]-DBx[k-1][j][0])/(MG->dz[k]+MG->dz[k-1]);
      //Curyh[k][j][L]+=(dt1+dt2)/miu*(DBx[k][j][L]-DBx[k-1][j][L])/(MG->dz[k]+MG->dz[k-1]);
      Curyh[k][j][0]+=(dt1+dt2)/miu*((DBx[k][j][0]-DBx[k-1][j][0])/(MG->dz[k]+MG->dz[k-1])-(DBz[k][j][0])/(2*MG->dx[0]));
      Curyh[k][j][L]+=(dt1+dt2)/miu*((DBx[k][j][L]-DBx[k-1][j][L])/(MG->dz[k]+MG->dz[k-1])-(-DBz[k][j][L-1])/(2*MG->dx[L-1]));
    }
   }
   
  #pragma omp parallel for collapse(2) private(k,j,i)
  for(k=1;k<N;k++)
  {  
    for(j=0;j<M;j++)
    {
      for(i=1;i<L;i++)
      {
        Curyh[k][j][i]+=(dt1+dt2)/miu*((DBx[k][j][i]-DBx[k-1][j][i])/(MG->dz[k]+MG->dz[k-1])-(DBz[k][j][i]-DBz[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]));
      }
    }
  }
	
  #pragma omp parallel for collapse(2) private(j,i)	
  for(j=0;j<M;j++)
  {
    for(i=1;i<L;i++)
    {
      //Curyh[N][j][i]-=(dt1+dt2)/miu*(DBz[N][j][i]-DBz[N][j][i-1])/(MG->dx[i]+MG->dx[i-1]);
      Curyh[N][j][i]+=(dt1+dt2)/miu*((-DBx[N-1][j][i])/(2*MG->dz[N-1])-(DBz[N][j][i]-DBz[N][j][i-1])/(MG->dx[i]+MG->dx[i-1]));
    }
  }

  ///curzh
  #pragma omp parallel for collapse(2) private(k,j,i,term1,term2,term3,Coeff)
  for(k=0;k<N;k++)
  {
    for(j=0;j<M+1;j++)
    {
      for(i=0;i<L+1;i++)
      {
        if(k==0)
        {
          term1=0;
          Coeff=MG->dz[0];
        }
        else
        {
          term1=Curzh[k-1][j][i];
          Coeff=MG->dz[k]+MG->dz[k-1];
        }


        if(j==0)
          term2=(Curyh[k][j][i])/(2*MG->dy[j]);
        else if(j==M)
          term2=(-Curyh[k][j-1][i])/(2*MG->dy[j-1]);
        else
          term2=(Curyh[k][j][i]-Curyh[k][j-1][i])/(MG->dy[j]+MG->dy[j-1]);

        if(i==0)
          term3=(Curxh[k][j][i])/(2*MG->dx[i]);
        else if(i==L)
          term3=(-Curxh[k][j][i-1])/(2*MG->dx[i-1]);
      /*  else if(i==MP->ScXa && j==MP->ScYa)
          term3=(Curxh[k][j][i]-Curxh[k][j][i-1]+arfa*(-Current))/(MG->dx[i]+MG->dx[i-1]);
        else if(i==MP->ScXb && j==MP->ScYb)
          term3=(Curxh[k][j][i]-Curxh[k][j][i-1]+arfa*(Current))/(MG->dx[i]+MG->dx[i-1]);
        */
        else
          term3=(Curxh[k][j][i]-Curxh[k][j][i-1])/(MG->dx[i]+MG->dx[i-1]);

        Curzh[k][j][i]=term1-Coeff*(term2+term3);

      }
    }
  }


  return 0;

}
