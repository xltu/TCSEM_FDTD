#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<fftw3.h>
double min(double A,double B)
{
  double C=A;
  if(A>B)
  C=B;
  return C;
}
double max(double A,double B)
{
  double C=A;
  if(A<B)
  C=B;
  return C;
}

int Free_3D_Array(double ***P,int Nz)
{
	int i,j;
	if(P!=NULL&&P[0][0]!=NULL)
	{
		free(P[0][0]);
		for(i=0;i<Nz;i++)
			free(P[i]);
		free(P);
		return 0;
	}
	else
		return 1;

}


double ***Create_3D_Array(int Nz,int Ny,int Nx)
{
	int i,j;
	double ***P;
	P=(double ***)malloc(Nz*sizeof(double **));
	for(i=0;i<Nz;i++)
		P[i]=(double **)malloc(Ny*sizeof(double *));

	P[0][0]=(double *)malloc(Nz*Ny*Nx*sizeof(double));

	for(i=0;i<Nz;i++)
	{
		for(j=0;j<Ny;j++)
			P[i][j]=P[0][0]+i*Ny*Nx+j*Nx;
	}

	memset(P[0][0],0,Nz*Ny*Nx*sizeof(double));

	if(P==NULL||P[0][0]==NULL)
	{
		printf("fail to allocate a 3D array sizeof Nz*Ny*Nx=%d*%d*%d\n",Nz,Ny,Nx);
		printf("press any key to get exit\n");
		getchar();
		exit(0);
	}
	return P;
}

double **Create2DArray(int M,int N)
{
	int i;
	double **Array;
	Array=(double **)malloc(M*sizeof(double *));
	Array[0]=(double *)malloc(M*N*sizeof(double));
	for(i=1;i<M;i++)
		Array[i]=Array[i-1]+N;
	memset(Array[0],0,M*N*sizeof(double));
	if(Array==NULL||Array[0]==NULL)
	{
		printf("fail to allocate a 2D array sizeof Ny*Nx=%d*%d\n",M,N);
		printf("press any key to get exit\n");
		getchar();
		exit(0);
	}
	return Array;
}

fftw_complex **Create2DfftwArray(int M,int N)
{
	int i;
	fftw_complex **Array;

	Array=(fftw_complex **)fftw_malloc(M*sizeof(fftw_complex *));
	Array[0]=(fftw_complex *)fftw_malloc(M*N*sizeof(fftw_complex));
	for(i=1;i<M;i++)
		Array[i]=Array[i-1]+N;
	memset(Array[0],0,M*N*sizeof(fftw_complex));
	if(Array==NULL||Array[0]==NULL)
	{
		printf("fail to allocate a 2D fftw_complex array sizeof Ny*Nx=%d*%d\n",M,N);
		printf("press any key to get exit\n");
		getchar();
		exit(0);
	}
	return Array;
}
//******************************************
//Function: Free2DArray
void Free2DArray(double **Array)
{
	free(Array[0]);
	free(Array);
}

// Function:: Free2DfftwArray
void Free2DfftwArray(fftw_complex **Array)
{
	fftw_free(Array[0]);
	fftw_free(Array);
}




