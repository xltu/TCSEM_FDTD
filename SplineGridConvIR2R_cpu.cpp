#include <cassert>
#include <vector>
#include <algorithm>
#include "FTDT.h"
#include "Spline.h"
#include "omp.h"
int SplineGridConvIR2R_cpu(ModelPara *MP,ModelGrid *MG,GridConv *GC,double **F_old,double **F_new)

/*int SplineGridConvIR2R(double *MG->dx,double *MG->dy,double **F_old,
							int irxL_old,int irxR_old,int iry_old,
							int irxL_New,int irxR_New,int iry_New,
							int Nx,int Ny,double *MG->X_Bzold,double *MG->Y_Bzold,
							double *MG->X_BzNew,double *MG->Y_BzNew,
							int *GC->xBzNew2old,int *GC->yBzNew2old,double **F_new)*/
/*
* F_old[M][L]   the uninterpolated value for the field
* F_new[M1][L1] the interpolated value in the new regular grid
* MG->X_Bzold[L]  	the X coordinates for Bz notes in the old grid
* MG->Y_Bzold[M]		the Y coordinates for Bz notes in the old grid
* MG->X_BzNew[L1]	the X ................................new grid
* MG->Y_BzNew[M1]	the Y ................................old grid
* GC->xBzNew2old[L1] the location mapping of Bz notes from new grid into old grid
* GC->yBzNew2old[M1] the location mapping of Bz notes from new grid into old grid
*/
{
	double **Semi;
	int irxL_old,irxR_old,irxL_New,irxR_New,iry_New,iry_old,Nx,Ny;
	irxL_old=MP->AnLx_old;
	irxL_New=MP->AnLx_New;
	irxR_old=MP->AnRx_old;
	irxR_New=MP->AnRx_New;
	iry_old=MP->Any_old;
	iry_New=MP->Any_New;
	Nx=MP->Nx;
	Ny=MP->Ny;

	int n1=irxL_old + 2;
	int n2=irxR_old + 2;
	int n3=iry_old + 2;
	tk::band_matrix A1(n1,1,1);
	tk::band_matrix A2(n2,1,1);
	tk::band_matrix A3(n3,1,1);

	std::vector<double>  rhs1(n1),rhs2(n2),rhs3(n3),m_a(n1),m_b(n1),m_c(n1);
	int i,j,id;
	
	#pragma omp parallel for collapse(2) private(i,j)
	for(j=0;j<Ny;j++)
	{
	  for(i=0;i<Nx;i++)
	  {
		  F_new[j+iry_New][i+irxL_New]=F_old[j+iry_old][i+irxL_old];
	  }
	}


	Semi = Create2DArray(irxL_New,iry_old + 2);

	for(i=1;i<n1-1;i++)
	{
		 A1(i,i-1)=1.0/3.0*MG->dx[i-1];
         A1(i,i)=2.0/3.0*(MG->dx[i]+MG->dx[i-1]);
         A1(i,i+1)=1.0/3.0*MG->dx[i];
   }
   A1(0,0)=2.0;
   A1(0,1)=0.0;
   A1(n1-1,n1-1)=2.0;
   A1(n1-1,n1-2)=0.0;

   ///x direction Cov of left-front cornor
   for(i=1;i<n1-1;i++)
   	rhs1[i]=(F_old[0][i+1]-F_old[0][i])/MG->dx[i] - (F_old[0][i]-F_old[0][i-1])/MG->dx[i-1];
   rhs1[0]=0.;
   rhs1[n1-1]=0.;

   // solve the equation system to obtain the parameters b[]
   m_b=A1.lu_solve(rhs1,false);

   // calculate parameters a[] and c[] based on b[]
   m_a.resize(n1);
   m_c.resize(n1);
   for(i=0; i<n1-1; i++)
   {
      m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i];
      m_c[i]=(F_old[0][i+1]-F_old[0][i])/MG->dx[i]
             - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i];
   }
   // for the right boundary we define
   // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
   double h=MG->dx[n1-2];
   // m_b[n-1] is determined by the boundary condition
   m_a[n1-1]=0.0;
   m_c[n1-1]=3.0*m_a[n1-2]*h*h+2.0*m_b[n1-2]*h+m_c[n1-2];   // = f'_{n-2}(x_{n-1})

   for(i=0;i<irxL_New;i++)
   {
   	id=max(GC->xBzNew2old[i]-1,0);
   	h=MG->X_BzNew[i]-MG->X_Bzold[id];
   	id=GC->xBzNew2old[i]-1;
   	if(id<0)
   		Semi[i][0]=((m_b[0])*h + m_c[0])*h + F_old[0][0];
   	else if(id==n1-1)
   		Semi[i][0]=((m_b[n1-1])*h + m_c[n1-1])*h + F_old[0][id];
   	else
   		Semi[i][0]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[0][id];
   }

   for(j=1;j<iry_old;j++)
   {
   	for(i=1;i<n1-1;i++)
   		rhs1[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i] - (F_old[j][i]-F_old[j][i-1])/MG->dx[i-1];
		rhs1[0]=0.;
		rhs1[n1-1]=0.;

		m_b=A1.lu_solve(rhs1,true);

		for(i=0; i<n1-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i];
		   m_c[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i];
		}

		h=MG->dx[n1-2];
		m_a[n1-1]=0.0;
		m_c[n1-1]=3.0*m_a[n1-2]*h*h+2.0*m_b[n1-2]*h+m_c[n1-2];
		for(i=0;i<irxL_New;i++)
		{
			id=max(GC->xBzNew2old[i]-1,0);
			h=MG->X_BzNew[i]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i]-1;
			if(id<0)
				Semi[i][j]=((m_b[0])*h + m_c[0])*h + F_old[j][0];
			else if(id==n1-1)
				Semi[i][j]=((m_b[n1-1])*h + m_c[n1-1])*h + F_old[j][id];
			else
				Semi[i][j]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id];
		}
	}

	///left
	for(j=iry_old;j<iry_old+Ny;j++)
	{
		for(i=1;i<n1-1;i++)
   		rhs1[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i] - (F_old[j][i]-F_old[j][i-1])/MG->dx[i-1];
		rhs1[0]=0.;
		rhs1[n1-1]=0.;

		m_b=A1.lu_solve(rhs1,true);

		for(i=0; i<n1-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i];
		   m_c[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i];
		}

		h=MG->dx[n1-2];
		m_a[n1-1]=0.0;
   	m_c[n1-1]=3.0*m_a[n1-2]*h*h+2.0*m_b[n1-2]*h+m_c[n1-2];
   	
   	
   	for(i=0;i<irxL_New;i++)
		{
			id=max(GC->xBzNew2old[i]-1,0);
			h=MG->X_BzNew[i]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i]-1;
			if(id<0)
				F_new[j-iry_old+iry_New][i]=((m_b[0])*h + m_c[0])*h + F_old[j][0];
			else if(id==n1-1)
				F_new[j-iry_old+iry_New][i]=((m_b[n1-1])*h + m_c[n1-1])*h + F_old[j][id];
			else
				F_new[j-iry_old+iry_New][i]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id];
		}
	}

	for(j=iry_old;j<iry_old + 2;j++)
		for(i=0;i<irxL_New;i++)
			Semi[i][j]=F_new[j-iry_old+iry_New][i];

	///y direction interpolation of the left-front cornor
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i-1];
      A3(i,i)=2.0/3.0*(MG->dy[i]+MG->dy[i-1]);
      A3(i,i+1)=1.0/3.0*MG->dy[i];
   }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;
   for(i=1;i<n3-1;i++)
   	rhs3[i]=(Semi[0][i+1]-Semi[0][i])/MG->dy[i] - (Semi[0][i]-Semi[0][i-1])/MG->dy[i-1];
   rhs3[0]=0.;
   rhs3[n3-1]=0.;

   m_b=A3.lu_solve(rhs3,false);
   m_a.resize(n3);
   m_c.resize(n3);

   for(i=0; i<n3-1; i++)
   {
      m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i];
      m_c[i]=(Semi[0][i+1]-Semi[0][i])/MG->dy[i]
             - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i];
   }

   h=MG->dy[n3-2];
   m_a[n3-1]=0.0;
   m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];

   for(i=0;i<iry_New;i++)
   {
   	id=max(GC->yBzNew2old[i]-1,0);
   	h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
   	id=GC->yBzNew2old[i]-1;
   	if(id<0)
   		F_new[i][0]=((m_b[0])*h + m_c[0])*h + Semi[0][0];
   	else if(id==n3-1)
   		F_new[i][0]=((m_b[n3-1])*h + m_c[n3-1])*h + Semi[0][id];
   	else
   		F_new[i][0]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + Semi[0][id];
   }

   for(j=1;j<irxL_New;j++)
   {
   	for(i=1;i<n3-1;i++)
   		rhs3[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i] - (Semi[j][i]-Semi[j][i-1])/MG->dy[i-1];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;

		m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i];
		   m_c[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i];
		}

		h=MG->dy[n3-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=0;i<iry_New;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			if(id<0)
				F_new[i][j]=((m_b[0])*h + m_c[0])*h + Semi[j][0];
			else if(id==n3-1)
				F_new[i][j]=((m_b[n3-1])*h + m_c[n3-1])*h + Semi[j][id];
			else
				F_new[i][j]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + Semi[j][id];
		}
	}


	///x direction left back
	m_a.resize(n1);
   m_c.resize(n1);
   for(j=iry_old+Ny-2;j<iry_old+Ny;j++)
   	for(i=0;i<irxL_New;i++)
   		Semi[i][j+2-iry_old-Ny]=F_new[j-iry_old+iry_New][i];
   for(j=iry_old+Ny;j<Ny+2*iry_old;j++)
   {
   	for(i=1;i<n1-1;i++)
   		rhs1[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i] - (F_old[j][i]-F_old[j][i-1])/MG->dx[i-1];
		rhs1[0]=0.;
		rhs1[n1-1]=0.;

		m_b=A1.lu_solve(rhs1,true);

		for(i=0; i<n1-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i];
		   m_c[i]=(F_old[j][i+1]-F_old[j][i])/MG->dx[i]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i];
		}

		h=MG->dx[n1-2];
		m_a[n1-1]=0.0;
   	m_c[n1-1]=3.0*m_a[n1-2]*h*h+2.0*m_b[n1-2]*h+m_c[n1-2];
   	for(i=0;i<irxL_New;i++)
		{
			id=max(GC->xBzNew2old[i]-1,0);
			h=MG->X_BzNew[i]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i]-1;
			if(id<0)
				Semi[i][j+2-iry_old-Ny]=((m_b[0])*h + m_c[0])*h + F_old[j][0];
			else if(id==n1-1)
				Semi[i][j+2-iry_old-Ny]=((m_b[n1-1])*h + m_c[n1-1])*h + F_old[j][id];
			else
				Semi[i][j+2-iry_old-Ny]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id];
		}
	}

	///y direction left back
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i+iry_old+Ny-3];
      A3(i,i)=2.0/3.0*(MG->dy[i+iry_old+Ny-2]+MG->dy[i+iry_old+Ny-3]);
      A3(i,i+1)=1.0/3.0*MG->dy[i+iry_old+Ny-2];
   }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;

   m_a.resize(n3);
   m_c.resize(n3);
   for(j=0;j<irxL_New;j++)
   {
   	for(i=1;i<n3-1;i++)
   		rhs3[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i+iry_old+Ny-2] - (Semi[j][i]-Semi[j][i-1])/MG->dy[i+iry_old+Ny-3];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;
		if(j==0)
			m_b=A3.lu_solve(rhs3,false);
		else
			m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i+iry_old+Ny-2];
		   m_c[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i+iry_old+Ny-2]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i+iry_old+Ny-2];
		}

		h=MG->dy[iry_old+Ny-2+n3-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=iry_New+Ny;i<2*iry_New+Ny;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			id-=(iry_old+Ny-2);
			if(id<0)
				F_new[i][j]=((m_b[0])*h + m_c[0])*h + Semi[j][0];
			else if(id==n3-1)
				F_new[i][j]=((m_b[n3-1])*h + m_c[n3-1])*h + Semi[j][id];
			else
				F_new[i][j]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + Semi[j][id];
		}
	}

	///front only y direction needed
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i-1];
      A3(i,i)=2.0/3.0*(MG->dy[i]+MG->dy[i-1]);
      A3(i,i+1)=1.0/3.0*MG->dy[i];
   }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;

	for(j=irxL_old;j<irxL_old+Nx;j++)
	{
		for(i=1;i<n3-1;i++)
   		rhs3[i]=(F_old[i+1][j]-F_old[i][j])/MG->dy[i] - (F_old[i][j]-F_old[i-1][j])/MG->dy[i-1];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;
		if(j==irxL_old)
			m_b=A3.lu_solve(rhs3,false);
		else
			m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i];
		   m_c[i]=(F_old[i+1][j]-F_old[i][j])/MG->dy[i]-1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i];
		}

		h=MG->dy[n3-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=0;i<iry_New;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			if(id<0)
				F_new[i][j-irxL_old+irxL_New]=((m_b[0])*h + m_c[0])*h + F_old[0][j];
			else if(id==n3-1)
				F_new[i][j-irxL_old+irxL_New]=((m_b[n3-1])*h + m_c[n3-1])*h + F_old[id][j];
			else
				F_new[i][j-irxL_old+irxL_New]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[id][j];
		}
	}

	///back only y direction
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i+iry_old+Ny-3];
      A3(i,i)=2.0/3.0*(MG->dy[i+iry_old+Ny-2]+MG->dy[i+iry_old+Ny-3]);
      A3(i,i+1)=1.0/3.0*MG->dy[i+iry_old+Ny-2];
   }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;

   for(j=irxL_old;j<irxL_old+Nx;j++)
	{
		for(i=1;i<n3-1;i++)
   		rhs3[i]=(F_old[i+1+iry_old+Ny-2][j]-F_old[i+iry_old+Ny-2][j])/MG->dy[i+iry_old+Ny-2]
   		- (F_old[i+iry_old+Ny-2][j]-F_old[i-1+iry_old+Ny-2][j])/MG->dy[i-1+iry_old+Ny-2];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;
		if(j==irxL_old)
			m_b=A3.lu_solve(rhs3,false);
		else
			m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i+iry_old+Ny-2];
		   m_c[i]=(F_old[i+1+iry_old+Ny-2][j]-F_old[i+iry_old+Ny-2][j])/MG->dy[i+iry_old+Ny-2]-1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i+iry_old+Ny-2];
		}

		h=MG->dy[n3-2+iry_old+Ny-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=iry_New+Ny;i<2*iry_New+Ny;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			id-=(Ny+iry_old-2);
			if(id<0)
				F_new[i][j-irxL_old+irxL_New]=((m_b[0])*h + m_c[0])*h + F_old[Ny+iry_old-2][j];
			else if(id==n3-1)
				F_new[i][j-irxL_old+irxL_New]=((m_b[n3-1])*h + m_c[n3-1])*h + F_old[id+Ny+iry_old-2][j];
			else
				F_new[i][j-irxL_old+irxL_New]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[id+Ny+iry_old-2][j];
		}
	}


	Free2DArray(Semi);

	Semi=Create2DArray(irxR_New,iry_old + 2);

	///right front x direction interpolation
	for(i=1;i<n2-1;i++)
	{
		   A2(i,i-1)=1.0/3.0*MG->dx[i+Nx+irxL_old-3];
         A2(i,i)=2.0/3.0*(MG->dx[i+Nx+irxL_old-2]+MG->dx[i+Nx+irxL_old-3]);
         A2(i,i+1)=1.0/3.0*MG->dx[i+Nx+irxL_old-2];
   }
  	A2(0,0)=2.0;
   A2(0,1)=0.0;
   A2(n2-1,n2-1)=2.0;
   A2(n2-1,n2-2)=0.0;

   m_a.resize(n2);
   m_c.resize(n2);
   for(j=0;j<iry_old;j++)
   {
   	for(i=1;i<n2-1;i++)
   		rhs2[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
   			- (F_old[j][i+Nx+irxL_old-2]-F_old[j][i+Nx+irxL_old-3])/MG->dx[i+Nx+irxL_old-3];
		rhs2[0]=0.;
		rhs2[n2-1]=0.;
		if(j==0)
			m_b=A2.lu_solve(rhs2,false);
		else
			m_b=A2.lu_solve(rhs2,true);

		for(i=0; i<n2-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i+Nx+irxL_old-2];
		   m_c[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i+Nx+irxL_old-2];
		}

		h=MG->dx[n2-2+Nx+irxL_old-2];
		m_a[n2-1]=0.0;
   	m_c[n2-1]=3.0*m_a[n2-2]*h*h+2.0*m_b[n2-2]*h+m_c[n2-2];
   	for(i=0;i<irxR_New;i++)
		{
			id=max(GC->xBzNew2old[i+Nx+irxL_New]-1,0);
			h=MG->X_BzNew[i+Nx+irxL_New]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i+Nx+irxL_New]-1;
			id-=(Nx+irxL_old-2);
			if(id<0)
				Semi[i][j]=((m_b[0])*h + m_c[0])*h + F_old[j][Nx+irxL_old-2];
			else if(id==n2-1)
				Semi[i][j]=((m_b[n2-1])*h + m_c[n2-1])*h + F_old[j][id+Nx+irxL_old-2];
			else
				Semi[i][j]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id+Nx+irxL_old-2];
		}
	}

	///right x direction
	for(j=iry_old;j<iry_old+Ny;j++)
	{
		for(i=1;i<n2-1;i++)
   		rhs2[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
   			- (F_old[j][i+Nx+irxL_old-2]-F_old[j][i+Nx+irxL_old-3])/MG->dx[i+Nx+irxL_old-3];
		rhs2[0]=0.;
		rhs2[n2-1]=0.;

		m_b=A2.lu_solve(rhs2,true);

		for(i=0; i<n2-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i+Nx+irxL_old-2];
		   m_c[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i+Nx+irxL_old-2];
		}

		h=MG->dx[n2-2+Nx+irxL_old-2];
		m_a[n2-1]=0.0;
   	m_c[n2-1]=3.0*m_a[n2-2]*h*h+2.0*m_b[n2-2]*h+m_c[n2-2];

		for(i=0;i<irxR_New;i++)
		{
			id=max(GC->xBzNew2old[i+Nx+irxL_New]-1,0);
			h=MG->X_BzNew[i+Nx+irxL_New]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i+Nx+irxL_New]-1;
			id-=(Nx+irxL_old-2);
			if(id<0)
				F_new[j-iry_old+iry_New][i+Nx+irxL_New]=((m_b[0])*h + m_c[0])*h + F_old[j][Nx+irxL_old-2];
			else if(id==n2-1)
				F_new[j-iry_old+iry_New][i+Nx+irxL_New]=((m_b[n2-1])*h + m_c[n2-1])*h + F_old[j][id+Nx+irxL_old-2];
			else
				F_new[j-iry_old+iry_New][i+Nx+irxL_New]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id+Nx+irxL_old-2];
		}
	}

	for(j=iry_old;j<iry_old + 2;j++)
		for(i=0;i<irxR_New;i++)
			Semi[i][j]=F_new[j-iry_old+iry_New][i+Nx+irxL_New-2];

	///right front y direction
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i-1];
      A3(i,i)=2.0/3.0*(MG->dy[i]+MG->dy[i-1]);
      A3(i,i+1)=1.0/3.0*MG->dy[i];
   }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;
   m_a.resize(n3);
   m_c.resize(n3);
   for(j=0;j<irxR_New;j++)
   {
   	for(i=1;i<n3-1;i++)
   		rhs3[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i] - (Semi[j][i]-Semi[j][i-1])/MG->dy[i-1];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;

		if(j==0)
			m_b=A3.lu_solve(rhs3,false);
		else
			m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i];
		   m_c[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i];
		}

		h=MG->dy[n3-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=0;i<iry_New;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			if(id<0)
				F_new[i][j+irxL_New+Nx]=((m_b[0])*h + m_c[0])*h + Semi[j][0];
			else if(id==n3-1)
				F_new[i][j+irxL_New+Nx]=((m_b[n3-1])*h + m_c[n3-1])*h + Semi[j][id];
			else
				F_new[i][j+irxL_New+Nx]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + Semi[j][id];
		}
	}

	///x direction right back
	m_a.resize(n2);
   m_c.resize(n2);
   for(j=iry_old+Ny-2;j<iry_old+Ny;j++)
   	for(i=0;i<irxR_New;i++)
   		Semi[i][j+2-iry_old-Ny]=F_new[j-iry_old+iry_New][i+Nx+irxL_New-2];

   for(j=iry_old+Ny;j<Ny+2*iry_old;j++)
   {
   	for(i=1;i<n2-1;i++)
   		rhs2[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
   		- (F_old[j][i+Nx+irxL_old-2]-F_old[j][i+Nx+irxL_old-3])/MG->dx[i+Nx+irxL_old-3];
		rhs2[0]=0.;
		rhs2[n2-1]=0.;

		//if(j==iry_old+Ny)
			//m_b=A2.lu_solve(rhs2,false);
		//else
			m_b=A2.lu_solve(rhs2,true);

		for(i=0; i<n2-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dx[i+Nx+irxL_old-2];
		   m_c[i]=(F_old[j][i+Nx+irxL_old-1]-F_old[j][i+Nx+irxL_old-2])/MG->dx[i+Nx+irxL_old-2]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dx[i+Nx+irxL_old-2];
		}

		h=MG->dx[n2-2+Nx+irxL_old-2];
		m_a[n2-1]=0.0;
   	m_c[n2-1]=3.0*m_a[n2-2]*h*h+2.0*m_b[n2-2]*h+m_c[n2-2];

   	for(i=0;i<irxR_New;i++)
		{
			id=max(GC->xBzNew2old[i+Nx+irxL_New]-1,0);
			h=MG->X_BzNew[i+Nx+irxL_New]-MG->X_Bzold[id];
			id=GC->xBzNew2old[i+Nx+irxL_New]-1;
			id-=(irxL_old+Nx-2);
			if(id<0)
				Semi[i][j+2-iry_old-Ny]=((m_b[0])*h + m_c[0])*h + F_old[j][irxL_old+Nx-2];
			else if(id==n1-1)
				Semi[i][j+2-iry_old-Ny]=((m_b[n2-1])*h + m_c[n2-1])*h + F_old[j][id+irxL_old+Nx-2];
			else
				Semi[i][j+2-iry_old-Ny]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + F_old[j][id+irxL_old+Nx-2];
		}
	}

	///y direction right back
	for(i=1;i<n3-1;i++)
	{
		A3(i,i-1)=1.0/3.0*MG->dy[i+iry_old+Ny-3];
    A3(i,i)=2.0/3.0*(MG->dy[i+iry_old+Ny-2]+MG->dy[i+iry_old+Ny-3]);
    A3(i,i+1)=1.0/3.0*MG->dy[i+iry_old+Ny-2];
  }
  	A3(0,0)=2.0;
   A3(0,1)=0.0;
   A3(n3-1,n3-1)=2.0;
   A3(n3-1,n3-2)=0.0;

   m_a.resize(n3);
   m_c.resize(n3);
   for(j=0;j<irxR_New;j++)
   {
   	for(i=1;i<n3-1;i++)
   		rhs3[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i+iry_old+Ny-2] - (Semi[j][i]-Semi[j][i-1])/MG->dy[i+iry_old+Ny-3];
		rhs3[0]=0.;
		rhs3[n3-1]=0.;
		if(j==0)
			m_b=A3.lu_solve(rhs3,false);
		else
			m_b=A3.lu_solve(rhs3,true);

		for(i=0; i<n3-1; i++)
		{
		   m_a[i]=1.0/3.0*(m_b[i+1]-m_b[i])/MG->dy[i+iry_old+Ny-2];
		   m_c[i]=(Semi[j][i+1]-Semi[j][i])/MG->dy[i+iry_old+Ny-2]
		          - 1.0/3.0*(2.0*m_b[i]+m_b[i+1])*MG->dy[i+iry_old+Ny-2];
		}

		h=MG->dy[iry_old+Ny-2+n3-2];
		m_a[n3-1]=0.0;
   	m_c[n3-1]=3.0*m_a[n3-2]*h*h+2.0*m_b[n3-2]*h+m_c[n3-2];
   	for(i=iry_New+Ny;i<2*iry_New+Ny;i++)
		{
			id=max(GC->yBzNew2old[i]-1,0);
			h=MG->Y_BzNew[i]-MG->Y_Bzold[id];
			id=GC->yBzNew2old[i]-1;
			id-=(iry_old+Ny-2);
			if(id<0)
				F_new[i][j+irxL_New+Nx]=((m_b[0])*h + m_c[0])*h + Semi[j][0];
			else if(id==n3-1)
				F_new[i][j+irxL_New+Nx]=((m_b[n3-1])*h + m_c[n3-1])*h + Semi[j][id];
			else
				F_new[i][j+irxL_New+Nx]=((m_a[id]*h + m_b[id])*h + m_c[id])*h + Semi[j][id];
		}
	}


	Free2DArray(Semi);



	return 0;

}
