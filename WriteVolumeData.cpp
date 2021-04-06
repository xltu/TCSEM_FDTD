#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"FTDT.h"
int WriteVolumeData(Efield *EF,Bfield *BF,ModelPara *MP,double toT,double dt,double sita_t)
{
  FILE *Fp;
  int L,M,N;
  double t1=0.5;
  double t2=1.;
  double t3=2;

  L=MP->Nx+MP->AnLx_old+MP->AnRx_old;
  M=MP->Ny+2*MP->Any_old;
  N=MP->Nz+MP->Anz;

  if(fabs(toT-3*sita_t-t1)<dt)
  {
    if((Fp=fopen("ExVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the ExVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ex[0][0],sizeof(double),(N+1)*(M+1)*L,Fp);

    fclose(Fp);

    ///Ey
    if((Fp=fopen("EyVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the EyVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ey[0][0],sizeof(double),(N+1)*M*(L+1),Fp);

    fclose(Fp);

    ///Ez
    if((Fp=fopen("EzVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the EzVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ez[0][0],sizeof(double),N*(M+1)*(L+1),Fp);

    fclose(Fp);

    ///Bx
    if((Fp=fopen("BxVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the BxVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bx[0][0],sizeof(double),N*M*(L+1),Fp);

    fclose(Fp);

    ///By
    if((Fp=fopen("ByVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the ByVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->By[0][0],sizeof(double),N*(M+1)*L,Fp);

    fclose(Fp);

    ///Bz
    if((Fp=fopen("BzVolume_time1.dat","wb"))==NULL)
    {
      printf("fail to open the BzVolume_time1.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bz[0][0],sizeof(double),(N+1)*M*L,Fp);

    fclose(Fp);

  }
  else if(fabs(toT-3*sita_t-t2)<dt)
  {
    if((Fp=fopen("ExVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the ExVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ex[0][0],sizeof(double),(N+1)*(M+1)*L,Fp);

    fclose(Fp);

    ///Ey
    if((Fp=fopen("EyVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the EyVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ey[0][0],sizeof(double),(N+1)*M*(L+1),Fp);

    fclose(Fp);

    ///Ez
    if((Fp=fopen("EzVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the EzVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ez[0][0],sizeof(double),N*(M+1)*(L+1),Fp);

    fclose(Fp);

    ///Bx
    if((Fp=fopen("BxVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the BxVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bx[0][0],sizeof(double),N*M*(L+1),Fp);

    fclose(Fp);

    ///By
    if((Fp=fopen("ByVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the ByVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->By[0][0],sizeof(double),N*(M+1)*L,Fp);

    fclose(Fp);

    ///Bz
    if((Fp=fopen("BzVolume_time2.dat","wb"))==NULL)
    {
      printf("fail to open the BzVolume_time2.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bz[0][0],sizeof(double),(N+1)*M*L,Fp);

    fclose(Fp);

  }
  else if(fabs(toT-3*sita_t-t3)<dt)
  {
    if((Fp=fopen("ExVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the ExVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ex[0][0],sizeof(double),(N+1)*(M+1)*L,Fp);

    fclose(Fp);

    ///Ey
    if((Fp=fopen("EyVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the EyVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ey[0][0],sizeof(double),(N+1)*M*(L+1),Fp);

    fclose(Fp);

    ///Ez
    if((Fp=fopen("EzVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the EzVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(EF->Ez[0][0],sizeof(double),N*(M+1)*(L+1),Fp);

    fclose(Fp);

    ///Bx
    if((Fp=fopen("BxVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the BxVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bx[0][0],sizeof(double),N*M*(L+1),Fp);

    fclose(Fp);

    ///By
    if((Fp=fopen("ByVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the ByVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->By[0][0],sizeof(double),N*(M+1)*L,Fp);

    fclose(Fp);

    ///Bz
    if((Fp=fopen("BzVolume_time3.dat","wb"))==NULL)
    {
      printf("fail to open the BzVolume_time3.dat file!\n");
      getchar();
      exit(1);
    }
    fwrite(&(MP->Ny),sizeof(int),1,Fp);
    fwrite(&(MP->Nx),sizeof(int),1,Fp);
    fwrite(&(MP->Nz),sizeof(int),1,Fp);
    fwrite(&M,sizeof(int),1,Fp);
    fwrite(&L,sizeof(int),1,Fp);
    fwrite(&N,sizeof(int),1,Fp);
    fwrite(BF->Bz[0][0],sizeof(double),(N+1)*M*L,Fp);

    fclose(Fp);

  }
  else
    return 1;

  return 0;
}
