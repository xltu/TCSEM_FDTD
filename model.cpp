/* generate the conductivity model for FDTD EM modeling */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"FTDT.h"
extern double minsita;
void Model(int Nx,int Ny,int Nz,int AnLx,int AnRx,int Any,int Anz,int Xa,int Xb,int Ya,int Yb)
{
  double ***Con;
  double *X,*Y,*Z,*dx,*dy,*dz;
  FILE *Fp;
  int i,j,k;
  int L,M,N;
  double dxmin=50.,dymin=50.,dzmin=50.,arg1=1.1,arg2=1.25;
  double offset=4000;
  int AnLx_New,AnRx_New,Any_New;
  L=Nx+AnLx+AnRx;
  M=Ny+2*Any;
  N=Nz+Anz;

  /** allocate the memory needed*/
  Con=Create_3D_Array(N,M,L);
  X=(double *)malloc((L+1)*sizeof(double));
  Y=(double *)malloc((M+1)*sizeof(double));
  Z=(double *)malloc((N+1)*sizeof(double));
  dx=(double *)malloc((L)*sizeof(double));
  dy=(double *)malloc((M)*sizeof(double));
  dz=(double *)malloc((N)*sizeof(double));

  if(X==NULL||Y==NULL||Z==NULL||dx==NULL||dy==NULL||dz==NULL)
  {
    printf("fail to get the needed memory!press any key to get exit!\n");
    getchar();
    exit(1);
  }

	/**the model employed*/
  for(k=0;k<20;k++)
  {
 		for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=3.0;
		}
  }
 /* for(k=20;k<30;k++)
  {
    for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=minsita/20;
		}
  }
*/
  for(k=20;k<Nz;k++)
  {
    for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=1.0/10.0;
		}
  }

/*
	for(k=0;k<9;k++)
  {
 		for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=minsita;
		}
  }
  for(k=9;k<10;k++)
  {
    for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=1./15.;
		}
  }
  for(k=10;k<16;k++)
  {
    for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=1./40;
		}
  }
  for(k=16;k<Nz;k++)
  {
    for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=1./10;
		}
  }
*/
/*
  for(k=36;k<40;k++)
  {
    for(j=Any+(Ny-1)/2-15;j<Any+(Ny-1)/2+15;j++)
    {
      for(i=(Xa+Xb)/2+int(offset/dxmin+0.5)-15;i<(Xa+Xb)/2+int(offset/dxmin+0.5)+15;i++)
      {
        Con[k][j][i]=1./100;
      }
    }

  }
*/
/*
	for(k=0;k<Nz;k++)
	{
		for(j=Any;j<Any+Ny;j++)
		{
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=minsita;
		}
	}
*/

	// the air layer

	///the auxiliary absorbing boundary

	///bottom
	for(k=Nz;k<N;k++)
		for(j=Any;j<Any+Ny;j++)
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=Con[Nz-1][j][i];
	///front
	for(j=Any+Ny;j<2*Any+Ny;j++)
		for(k=0;k<N;k++)
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=Con[k][Any+Ny-1][i];
	///back
	for(j=0;j<Any;j++)
		for(k=0;k<N;k++)
			for(i=AnLx;i<AnLx+Nx;i++)
				Con[k][j][i]=Con[k][Any][i];

	///left
	for(i=0;i<AnLx;i++)
		for(k=0;k<N;k++)
			for(j=0;j<M;j++)
				Con[k][j][i]=Con[k][j][AnLx];
	///right
	for(i=AnLx+Nx;i<AnRx+AnLx+Nx;i++)
		for(k=0;k<N;k++)
			for(j=0;j<M;j++)
				Con[k][j][i]=Con[k][j][AnLx+Nx-1];
  //

  ///the temporal steps

 /* for(i=0;i<L;i++)
    dx[i]=10.;

  for(j=0;j<M;j++)
    dy[j]=10.;

  for(k=0;k<N;k++)
    dz[k]=10.; */

  for(i=AnLx;i<Nx+AnLx;i++)
    dx[i]=dxmin;
  for(i=AnLx-1;i>9;i--)
    dx[i]=dx[i+1]*arg1;
  for(i=9;i>=0;i--)
    dx[i]=dx[i+1]*arg2;
  for(i=Nx+AnLx;i<L-10;i++)
    dx[i]=dx[i-1]*arg1;
  for(i=L-10;i<L;i++)
    dx[i]=dx[i-1]*arg2;

  for(j=Any;j<Ny+Any;j++)
    dy[j]=dymin;
  for(j=Any-1;j>=10;j--)
    dy[j]=dy[j+1]*arg1;
  for(j=9;j>=0;j--)
    dy[j]=dy[j+1]*arg2;
  for(j=Ny+Any;j<M-10;j++)
    dy[j]=dy[j-1]*arg1;
  for(j=M-10;j<M;j++)
    dy[j]=dy[j-1]*arg2;

  for(k=0;k<Nz;k++)
    dz[k]=dzmin;
  for(k=Nz;k<N-10;k++)
    dz[k]=dz[k-1]*arg1;
  for(k=N-10;k<N;k++)
    dz[k]=dz[k-1]*arg2;

  ///the coordinate position of every point
  X[L/2-1]=0.;
  Y[M/2-1]=0.;
  Z[0]=0.;
  
  for(i=ceil(L/2)-2;i>=0;i--)
    X[i]=X[i+1]-dx[i];
  for(j=ceil(M/2)-2;j>=0;j--)
    Y[j]=Y[j+1]-dy[j];
    
  for(i=ceil(L/2);i<=L;i++)
    X[i]=X[i-1]+dx[i-1];
  for(j=ceil(M/2);j<=M;j++)
    Y[j]=Y[j-1]+dy[j-1];
    
  for(k=1;k<=N;k++)
    Z[k]=Z[k-1]+dz[k-1];

  ///the number of grids in the regular grid
  //AnLx_New=ceil((X[AnLx]+0.5*dx[0]+0.5*dxmin)/dxmin);
  //AnRx_New=ceil((X[L]+0.5*dx[L-1]-X[AnLx+Nx]+0.5*dxmin)/dxmin);
	//Any_New=ceil((Y[Any]+0.5*dy[0]+0.5*dymin)/dymin);
  AnLx_New=ceil((X[AnLx]-X[0])/dxmin);
  AnRx_New=ceil((X[L]-X[AnLx+Nx])/dxmin);
  Any_New=ceil((Y[Any]-Y[0])/dymin);

  /// save the model to a file model.dat
  if((Fp=fopen("model.dat","wb"))==NULL)
  {
    printf("fail to create the model file!\n");
    getchar();
    exit(1);
  }

  fwrite(&AnLx,sizeof(int),1,Fp);
  fwrite(&AnRx,sizeof(int),1,Fp);
  fwrite(&Nx,sizeof(int),1,Fp);
  fwrite(&Any,sizeof(int),1,Fp);
  fwrite(&Ny,sizeof(int),1,Fp);
  fwrite(&Anz,sizeof(int),1,Fp);
  fwrite(&Nz,sizeof(int),1,Fp);
  fwrite(&AnLx_New,sizeof(int),1,Fp);
  fwrite(&AnRx_New,sizeof(int),1,Fp);
  fwrite(&Any_New,sizeof(int),1,Fp);

  fwrite(dx,sizeof(double),AnLx+AnRx+Nx,Fp);
  fwrite(dy,sizeof(double),2*Any+Ny,Fp);
  fwrite(dz,sizeof(double),Anz+Nz,Fp);
  fwrite(X,sizeof(double),AnLx+AnRx+Nx+1,Fp);
  fwrite(Y,sizeof(double),2*Any+Ny+1,Fp);
  fwrite(Z,sizeof(double),Anz+Nz+1,Fp);
  fwrite(&Xa,sizeof(int),1,Fp);
  fwrite(&Xb,sizeof(int),1,Fp);
  fwrite(&Ya,sizeof(int),1,Fp);
  fwrite(&Yb,sizeof(int),1,Fp);

  fwrite(**Con,sizeof(double),N*M*L,Fp);
  fclose(Fp);

  Free_3D_Array(Con,N);
  free(X);
  free(Y);
  free(Z);
  free(dx);
  free(dy);
  free(dz);

  printf("the end of the model assemble\n");
}

