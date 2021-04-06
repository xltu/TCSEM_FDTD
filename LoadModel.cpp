/* this file used to load the model.dat binary file containing
 the model parameter and source parameters*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"FTDT.h"
int LoadModel(double ***Con,ModelPara *MP,ModelGrid *MG)
{
  FILE *Fp=NULL;
  int L,M,N,i,j,k;

  Model(140,120,70,40,40,40,40,68,71,100,100);

  Fp=fopen("model.dat","rb");

  if(NULL==Fp)
  {
    printf("fail to open the model data file!\n");
    getchar();
    exit(1);
  }

  fread(&(MP->AnLx_old),sizeof(int),1,Fp);
  fread(&(MP->AnRx_old),sizeof(int),1,Fp);
  fread(&(MP->Nx),sizeof(int),1,Fp);
  fread(&(MP->Any_old),sizeof(int),1,Fp);
  fread(&(MP->Ny),sizeof(int),1,Fp);
  fread(&(MP->Anz),sizeof(int),1,Fp);
  fread(&(MP->Nz),sizeof(int),1,Fp);
  fread(&(MP->AnLx_New),sizeof(int),1,Fp);
  fread(&(MP->AnRx_New),sizeof(int),1,Fp);
  fread(&(MP->Any_New),sizeof(int),1,Fp);

  L=MP->AnLx_old+MP->AnRx_old+MP->Nx;
  M=2*MP->Any_old+MP->Ny;
  N=MP->Nz+MP->Anz;

  fread((MG->dx),sizeof(double),L,Fp);
  fread((MG->dy),sizeof(double),M,Fp);
  fread((MG->dz),sizeof(double),N,Fp);

  fseek(Fp,(L+1+M+1+N+1)*sizeof(double),SEEK_CUR);

  fread(&(MP->ScXa),sizeof(int),1,Fp);
  fread(&(MP->ScXb),sizeof(int),1,Fp);
  fread(&(MP->ScYa),sizeof(int),1,Fp);
  fread(&(MP->ScYb),sizeof(int),1,Fp);

  fread(**Con,sizeof(double),N*M*L,Fp);

  fclose(Fp);

  MP->dxmin=MG->dx[MP->AnLx_old];
  MP->dymin=MG->dy[MP->Any_old];
  MP->dzmin=MG->dz[0];

  ///caculate the old coordinates
	MG->X_Bzold[L/2]=0.5*MG->dx[L/2];
	for(i=L/2+1;i<L;i++)
		MG->X_Bzold[i]=MG->X_Bzold[i-1]+0.5*MG->dx[i-1]+0.5*MG->dx[i];
	for(i=L/2-1;i>=0;i--)
		MG->X_Bzold[i]=MG->X_Bzold[i+1]-0.5*MG->dx[i+1]-0.5*MG->dx[i];
	
	MG->Y_Bzold[M/2]=0.5*MG->dy[M/2];
	for(j=M/2+1;j<M;j++)
		MG->Y_Bzold[j]=MG->Y_Bzold[j-1]+0.5*MG->dy[j-1]+0.5*MG->dy[j];	
	for(j=M/2-1;j>=0;j--)
		MG->Y_Bzold[j]=MG->Y_Bzold[j+1]-0.5*MG->dy[j+1]-0.5*MG->dy[j];
		
	MG->Z[0]=0.5*MG->dz[0];
	for(i=1;i<N;i++)
		MG->Z[i]=MG->Z[i-1]+0.5*MG->dz[i-1]+0.5*MG->dz[i];	
		
	/*		
	MG->X_Bzold[0]=0.5*MG->dx[0];
	for(i=1;i<L;i++)
		MG->X_Bzold[i]=MG->X_Bzold[i-1]+0.5*MG->dx[i-1]+0.5*MG->dx[i];
	MG->Y_Bzold[0]=0.5*MG->dy[0];
	for(j=1;j<M;j++)
		MG->Y_Bzold[j]=MG->Y_Bzold[j-1]+0.5*MG->dy[j-1]+0.5*MG->dy[j];
	*/	

  MP->LNx2= pow(2,floor (log(MP->AnLx_New+MP->AnRx_New+MP->Nx)/log(2.)) + 1);
  MP->LMy2= pow(2,floor (log(2*MP->Any_New+MP->Ny)/log(2.)) + 1);
	
  return 0;
}
