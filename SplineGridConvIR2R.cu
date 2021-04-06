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

/* The CPU code using a Natrual Cubic Spline interpolation, which needs to solve a 
*	 series of sparse linear systems, which is hard to implemented on GPU	
* int SplineGridConvIR2R(ModelPara *MP,ModelGrid *MG,GridConv *GC,double **F_old,double **F_new)
*
* F_old[M][L]   the uninterpolated value for the field
* F_new[M1][L1] the interpolated value in the new regular grid
* MG->X_Bzold[L]  	the X coordinates for Bz notes in the old grid
* MG->Y_Bzold[M]		the Y coordinates for Bz notes in the old grid
* MG->X_BzNew[L1]	the X ................................new grid
* MG->Y_BzNew[M1]	the Y ................................old grid
* GC->xBzNew2old[L1] the location mapping of Bz notes from new grid into old grid
* GC->yBzNew2old[M1] the location mapping of Bz notes from new grid into old grid
*
* MG->X_BzNew[i] lies in the left of MG->X_Bzold[0]      												GC->xBzNew2old[i]=0;
* MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]      	0<=i<MP->AnLx_New
* MG->X_BzNew[i] lies at the same point with MG->X_Bzold[GC->xBzNew2old[i]]     				 		MP->AnLx_New<=i<MP->AnLx_New+Nx
* MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]			MP->AnLx_New+Nx<=i<MP->AnLx_New+MP->AnRx_New+Nx
* MG->X_BzNew[i] lies in the right of MG->X_Bzold[L-1]                                       GC->xBzNew2old[i]=L;
*
* MG->Y_BzNew[j] lies in the front of MG->Y_Bzold[0]                         					GC->yBzNew2old[i]=0;
* MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]      	0<=i<MP->Any_New
* MG->Y_BzNew[j] lies at the same point as MG->Y_Bzold[GC->yBzNew2old[j]]              			MP->Any_New<=i<MP->Any_New+Ny
* MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]			MP->Any_New+Ny<=i<M1
* MG->Y_BzNew[j] lies in the back of MG->Y_Bzold[M-1]													 GC->yBzNew2old[i]=M;
*

*	XXX This GPU code use global natural cubic spline interpolation to interpolate the
*			Bz field in a regular dense mesh using the Bz field in the user defined irregular
*			coarse mesh
*
* REVIEW Corrections and Updates
*	XT @ 2021.02.25 Instead of padding only two points as in the interpolation in the previous
* 								version 20210216, this version padding more (NPad) points in the boundary.
*									 This is only for accuracy consideration. Seems not too much improvements.	
*/
 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "FTDT.h"
#include "Cuda_device.h"

__global__ void Fill_B_0x( realw *Bz, realw *B, const ModelPara *d_mp, 
	ModelGrid MG)
{
	volatile unsigned int ix = threadIdx.y * blockDim.x + threadIdx.x; // thread id inside block
	volatile unsigned int iy = blockIdx.x; 	// block id 
	volatile unsigned int n1=d_mp->ALx1 + d_mp->ALx2 + NPad;
	if( ix < n1 && iy < d_mp->M )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		sh_dx[ix] = MG.dx[ix];
		sh_bz[ix] = Bz[iy*d_mp->L+ix];
		__syncthreads( );
		if( ix==0 || ix == (n1-1) )
			B[iy*n1+ix]	=	zerof;	
		else
			B[iy*n1+ix]	=	(sh_bz[ix+1] - sh_bz[ix])/( 0.5*sh_dx[ix] + 0.5*sh_dx[ix+1]) 
									- (sh_bz[ix] - sh_bz[ix-1])/( 0.5*sh_dx[ix-1] + 0.5*sh_dx[ix]);
	}
}

__global__ void Fill_B_1x( realw *Bz, realw *B, const ModelPara *d_mp, 
	ModelGrid MG)
{
	volatile unsigned int ix = threadIdx.y * blockDim.x + threadIdx.x; // thread id inside block
	volatile unsigned int iy = blockIdx.x; 	// block id 
	volatile unsigned int n2=d_mp->ARx1 + d_mp->ARx2 + NPad;
	if( ix< n2 && iy < d_mp->M )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		sh_dx[ix] = MG.dx[d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + ix];
		sh_bz[ix] = Bz[iy*d_mp->L + d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + ix];
		__syncthreads( );
		if( ix==0 || ix == (n2-1) )
			B[iy*n2+ix]	=	zerof;	
		else
			B[iy*n2+ix]	=	(sh_bz[ix+1] - sh_bz[ix])/( 0.5*sh_dx[ix] + 0.5*sh_dx[ix+1]) 
									- (sh_bz[ix] - sh_bz[ix-1])/( 0.5*sh_dx[ix-1] + 0.5*sh_dx[ix]);
	}
}

// ANCHOR Filling B matrix in y direction interpolation requires the bz data from  three
// different matrix  (UPP->TDS0x).bz_semi0, Bz_old, and (UPP->TDS1x).bz_semi0; due to the data
// dependence, this kernal must be executed after the two interpolations in x direction
__global__ void Fill_B_0y( realw *Bold, realw *bz_semi_x00, realw *bz_semi_x10, realw *B, 
								const ModelPara *d_mp, ModelGrid MG)
{
	volatile unsigned int ix = threadIdx.y * blockDim.x + threadIdx.x; // thread id inside block
	volatile unsigned int iy = blockIdx.x; 	// block id 
	volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2 + NPad;
	if( ix< n3 && iy < d_mp->AnLx_New + d_mp->Nx + d_mp->AnRx_New )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		sh_dy[ix] = MG.dy[ix];
		if( iy<d_mp->AnLx_New ) 
			sh_bz[ix] = bz_semi_x00[ iy + ix * d_mp->AnLx_New ];
		else if( iy>= d_mp->AnLx_New && iy < d_mp->Nx + d_mp->AnLx_New )
			sh_bz[ix] = Bold[iy- d_mp->AnLx_New + d_mp->ALx1 + d_mp->ALx2 + ix*d_mp->L];			//REVIEW Bold index
		else
			sh_bz[ix] = bz_semi_x10[ iy - (d_mp->Nx + d_mp->AnLx_New) + ix * d_mp->AnRx_New ];

		__syncthreads( );
		if( ix==0 || ix == (n3-1) )
			B[iy*n3+ix]	=	zerof;	
		else
			B[iy*n3+ix]	=	(sh_bz[ix+1] - sh_bz[ix])/( 0.5*sh_dy[ix] + 0.5*sh_dy[ix+1]) 
									- (sh_bz[ix] - sh_bz[ix-1])/( 0.5*sh_dy[ix-1] + 0.5*sh_dy[ix]);
	}
}

// ANCHOR Filling B matrix in y direction interpolation requires the bz data from  three
// different matrix  (UPP->TDS0x).bz_semi1, Bz_old, and (UPP->TDS1x).bz_semi1; due to the data
// dependence, this kernal must be executed after the two interpolations in x direction
__global__ void Fill_B_1y( realw *Bold, realw *bz_semi_x01, realw *bz_semi_x11, realw *B, 
								const ModelPara *d_mp, ModelGrid MG)
{
	volatile unsigned int ix = threadIdx.y * blockDim.x + threadIdx.x; // thread id inside block
	volatile unsigned int iy = blockIdx.x; 	// block id 
	volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2 + NPad;
	if( ix< n3 && iy < d_mp->AnLx_New + d_mp->Nx + d_mp->AnRx_New )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];

		sh_dy[ix] = MG.dy[d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + ix];

		if( iy<d_mp->AnLx_New ) 
			sh_bz[ix] = bz_semi_x01[ iy + ix * d_mp->AnLx_New ];
		else if( iy>= d_mp->AnLx_New && iy < d_mp->Nx + d_mp->AnLx_New )
			sh_bz[ix] = Bold[iy- d_mp->AnLx_New + d_mp->ALx1 + d_mp->ALx2 
												+ (d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + ix)*d_mp->L]; //REVIEW same line
		else
			sh_bz[ix] = bz_semi_x11[ iy - (d_mp->Nx + d_mp->AnLx_New) + ix * d_mp->AnRx_New ];

		__syncthreads( );
		if( ix==0 || ix == (n3-1) )
			B[iy*n3+ix]	=	zerof;	
		else
			B[iy*n3+ix]	=	(sh_bz[ix+1] - sh_bz[ix])/( 0.5*sh_dy[ix] + 0.5*sh_dy[ix+1]) 
									- (sh_bz[ix] - sh_bz[ix-1])/( 0.5*sh_dy[ix-1] + 0.5*sh_dy[ix]);
	}
}

/* update coefficients ma,mb,mc and interpolation
* ANCHOR interpolation in the left part of the iregular grids
* MG->X_BzNew[i] lies in the left of MG->X_Bzold[0]      																								GC->xBzNew2old[i]=0;
* MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]      	0<=i<MP->AnLx_New
*/
__global__ void Intep_BzIR2R_0x( realw *Bold, realw *bz_semi_x00, realw *Bnew,
																 realw *bz_semi_x01, realw *B, const ModelPara *d_mp, 
																 ModelGrid MG, int *xBzNew2old )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // 
	
	if( ix < d_mp->AnLx_New && iy< d_mp->M )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		__shared__ realw sh_mb[BLOCKSIZE3];

		volatile unsigned int n1=d_mp->ALx1 + d_mp->ALx2 + NPad;

		if( threadIdx.x < n1 ) 
		{
			sh_dx[threadIdx.x] = MG.dx[threadIdx.x];
			sh_bz[threadIdx.x] = Bold[iy*d_mp->L+threadIdx.x];	// bz_old
			sh_mb[threadIdx.x] = B[iy*n1+threadIdx.x];					// mb
		}
		__syncthreads( );

		// REVIEW why do we need this???
		int id = max( xBzNew2old[ix]-1, 0);
#if MAXDEBUG == 1
		if(id > n1-1)
			printf("error in Intep_BzIR2R_0x\n");
#endif		
		realw ma, mc;
		if( id < n1-1 )
		{
			ma = 1.0/3.0*(sh_mb[id+1]-sh_mb[id])/(0.5*sh_dx[id]+0.5*sh_dx[id+1]);
			mc =(sh_bz[id+1]-sh_bz[id])/(0.5*sh_dx[id]+0.5*sh_dx[id+1])
		          - 1.0/3.0*(2.0*sh_mb[id]+sh_mb[id+1])*(0.5*sh_dx[id]+0.5*sh_dx[id+1]);
		}
		else
		{
			ma = 1.0/3.0*(sh_mb[id]-sh_mb[id-1])/(0.5*sh_dx[id-1]+0.5*sh_dx[id]);
			mc =(sh_bz[id]-sh_bz[id-1])/(0.5*sh_dx[id-1]+0.5*sh_dx[id])
		          - 1.0/3.0*(2.0*sh_mb[id-1]+sh_mb[id])*(0.5*sh_dx[id-1]+0.5*sh_dx[id]);

			mc += 3.0*ma*(0.5*sh_dx[id-1]+0.5*sh_dx[id])*(0.5*sh_dx[id-1]+0.5*sh_dx[id]) 
						+ 2.0*sh_mb[id-1]*(0.5*sh_dx[id-1]+0.5*sh_dx[id]);
			ma = zerof;
		}
		  
		realw h = MG.X_BzNew[ix] - MG.X_Bzold[id];
		id = xBzNew2old[ix]-1;
		realw fbz;
		if( id >= 0 && id < n1-1 )
			fbz = ma * h*h*h + sh_mb[id]*h*h + mc*h + sh_bz[id];
		else
			fbz = ( sh_mb[max(id,0)]*h + mc )*h + sh_bz[max(id,0)];

		// NOTE should check wheter should use if--else if--else branches here, prefer to use
		// multiple if here it seems its more efficient 
		// copy fbz to bz_semi_x00	
		if (iy < d_mp->Ay1 + d_mp->Ay2 + NPad )
			bz_semi_x00[iy*d_mp->AnLx_New + ix] = fbz;
		// copy to bz_new
		if (iy > d_mp->Ay1 + d_mp->Ay2 -1 && iy < ( d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny ) )
			Bnew[(iy - d_mp->Ay1 - d_mp->Ay2 + d_mp->Any_New)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx) + ix] = fbz;
		//copy to bz_semi_x01
		if ( iy > d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny - NPad -1 )
			bz_semi_x01[( iy - (d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny - NPad) )*d_mp->AnLx_New + ix] = fbz;
	}
}

/* ANCHOR interpolation in the right part of the iregular grids
* MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]			
*																																				MP->AnLx_New+Nx<=i<MP->AnLx_New+MP->AnRx_New+Nx
* MG->X_BzNew[i] lies in the right of MG->X_Bzold[L-1]                     GC->xBzNew2old[i]=L;
*/
__global__ void Intep_BzIR2R_1x( realw *Bold, realw *bz_semi_x10, realw *Bnew,
																 realw *bz_semi_x11, realw *B, const ModelPara *d_mp, 
																 ModelGrid MG, int *xBzNew2old )
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // 
	
	if( ix < d_mp->AnRx_New && iy< d_mp->M )
	{
		__shared__ realw sh_dx[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		__shared__ realw sh_mb[BLOCKSIZE3];

		volatile unsigned int n2 = d_mp->ARx1 + d_mp->ARx2 + NPad;

		if( threadIdx.x < n2 ) 
		{
			sh_dx[threadIdx.x] = MG.dx[d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + threadIdx.x];
			sh_bz[threadIdx.x] = Bold[iy*d_mp->L + d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + threadIdx.x];	// REVIEW bz_old
			sh_mb[threadIdx.x] = B[iy*n2+threadIdx.x];																										// mb
		}
		__syncthreads( );

		// REVIEW could be min(id, n2-1)??, id is garanteed to be id <= n2-1 
		int id = max( xBzNew2old[d_mp->Nx + d_mp->AnLx_New + ix] - 1 -(d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad), 0 );
#if MAXDEBUG == 1
		if(id < 0)
			printf("error in Intep_BzIR2R_1x\n");
#endif		
		realw ma, mc;
		if( id < n2-1 )
		{
			ma = 1.0/3.0*(sh_mb[id+1]-sh_mb[id])/(0.5*sh_dx[id]+0.5*sh_dx[id+1]);
			mc =(sh_bz[id+1]-sh_bz[id])/(0.5*sh_dx[id]+0.5*sh_dx[id+1])
							- 1.0/3.0*(2.0*sh_mb[id]+sh_mb[id+1])*(0.5*sh_dx[id]+0.5*sh_dx[id+1]);				
		}
		else
		{
			ma = 1.0/3.0*(sh_mb[id]-sh_mb[id-1])/(0.5*sh_dx[id-1]+0.5*sh_dx[id]);
			mc =(sh_bz[id]-sh_bz[id-1])/(0.5*sh_dx[id-1]+0.5*sh_dx[id])
		          - 1.0/3.0*(2.0*sh_mb[id-1]+sh_mb[id])*(0.5*sh_dx[id-1]+0.5*sh_dx[id]);

			mc += 3.0*ma*(0.5*sh_dx[id-1]+0.5*sh_dx[id])*(0.5*sh_dx[id-1]+0.5*sh_dx[id]) 
						+ 2.0*sh_mb[id-1]*(0.5*sh_dx[id-1]+0.5*sh_dx[id]);
			ma = zerof;
		}
		
		realw h = MG.X_BzNew[d_mp->Nx + d_mp->AnLx_New + ix] 
															- MG.X_Bzold[d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 - NPad + id];
		
		id = xBzNew2old[d_mp->Nx + d_mp->AnLx_New + ix] - 1 -(d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 -NPad);
		realw fbz;
		if( id >= 0 && id < n2-1 )
			fbz = ma * h*h*h + sh_mb[id]*h*h + mc*h + sh_bz[id];
		else
			fbz = ( sh_mb[max(id,0)]*h + mc )*h + sh_bz[max(id,0)];

		// REVIEW should check wheter should use if--else if--else branches here, prefer to use
		// multiple if here it seems its more efficient 
		// copy fbz to bz_semi_x00	
		if (iy < d_mp->Ay1 + d_mp->Ay2 + NPad )
			bz_semi_x10[iy*d_mp->AnRx_New + ix] = fbz;
		// copy to bz_new
		if (iy > d_mp->Ay1 + d_mp->Ay2 -1 && iy < d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny )
			Bnew[(iy - d_mp->Ay1 - d_mp->Ay2 + d_mp->Any_New)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx) + ix + d_mp->Nx + d_mp->AnLx_New] = fbz;
		//copy to bz_semi_x01
		if ( iy > d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny-NPad-1 )
			bz_semi_x11[( iy - (d_mp->Ay1 + d_mp->Ay2 + d_mp->Ny-NPad) )*d_mp->AnRx_New + ix] = fbz;
	}
}

/* update coefficients ma,mb,mc and interpolation
* ANCHOR interpolation in the front part of the iregular grids
* MG->Y_BzNew[j] lies in the front of MG->Y_Bzold[0]                         					GC->yBzNew2old[i]=0;
* MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]      	0<=i<MP->Any_New
*/
__global__ void Intep_BzIR2R_0y( realw *Bold, realw *bz_semi_x00, realw *Bnew,
																 realw *bz_semi_x10, realw *B, const ModelPara *d_mp, 
																 ModelGrid MG, int *yBzNew2old )
{
	// NOTE ix refers to y direction, iy represents x direction
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // 
	if( ix < d_mp->Any_New && iy< (d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx) )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		__shared__ realw sh_mb[BLOCKSIZE3];

		volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2+NPad;
		// copy from bz_semi_x00
		if( iy < d_mp->AnLx_New ) 
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[threadIdx.x];
				sh_bz[threadIdx.x] = bz_semi_x00[iy + threadIdx.x * d_mp->AnLx_New];
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];					// mb
			}
		}
		// copy from bz_old
		else if( iy >= d_mp->AnLx_New && iy < d_mp->AnLx_New + d_mp->Nx )
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[threadIdx.x];
				sh_bz[threadIdx.x] = Bold[iy- d_mp->AnLx_New + d_mp->ALx1 + d_mp->ALx2 + threadIdx.x*d_mp->L];		// REVIEW bz_old
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];							// mb
			}
		}	
		// copy from bz_semi_x10
		else
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[threadIdx.x];
				sh_bz[threadIdx.x] = bz_semi_x10[iy - (d_mp->Nx + d_mp->AnLx_New) + threadIdx.x * d_mp->AnRx_New];
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];					// mb
			}
		}
		__syncthreads( );

		// REVIEW why do we need this???
		int id = max( yBzNew2old[ix]-1, 0);
#if MAXDEBUG == 1
		if(id > n3-1)
			printf("error in Intep_BzIR2R_0y\n");
#endif		
		realw ma, mc;
		if( id < n3-1 )
		{
			ma = 1.0/3.0*(sh_mb[id+1]-sh_mb[id])/(0.5*sh_dy[id]+0.5*sh_dy[id+1]);
			mc =(sh_bz[id+1]-sh_bz[id])/(0.5*sh_dy[id]+0.5*sh_dy[id+1])
							- 1.0/3.0*(2.0*sh_mb[id]+sh_mb[id+1])*(0.5*sh_dy[id]+0.5*sh_dy[id+1]);						
		}
		else
		{
			ma = 1.0/3.0*(sh_mb[id]-sh_mb[id-1])/(0.5*sh_dy[id-1]+0.5*sh_dy[id]);
			mc =(sh_bz[id]-sh_bz[id-1])/(0.5*sh_dy[id-1]+0.5*sh_dy[id])
		          - 1.0/3.0*(2.0*sh_mb[id-1]+sh_mb[id])*(0.5*sh_dy[id-1]+0.5*sh_dy[id]);

			mc += 3.0*ma*(0.5*sh_dy[id-1]+0.5*sh_dy[id])*(0.5*sh_dy[id-1]+0.5*sh_dy[id]) 
						+ 2.0*sh_mb[id-1]*(0.5*sh_dy[id-1]+0.5*sh_dy[id]);
			ma = zerof;
		}
		  
		realw h = MG.Y_BzNew[ix] - MG.Y_Bzold[id];
		id = yBzNew2old[ix]-1;
		if( id >= 0 && id < n3-1 )
			Bnew[iy+ix*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] = ma * h*h*h + sh_mb[id]*h*h + mc*h + sh_bz[id];
		else
			Bnew[iy+ix*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] = ( sh_mb[max(id,0)]*h + mc )*h + sh_bz[max(id,0)];
	}
}
/* update coefficients ma,mb,mc and interpolation
* ANCHOR interpolation in the back part of the iregular grids
* MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]			MP->Any_New+Ny<=i<M1
* MG->Y_BzNew[j] lies in the back of MG->Y_Bzold[M-1]													 GC->yBzNew2old[i]=M;
*/
__global__ void Intep_BzIR2R_1y( realw *Bold, realw *bz_semi_x01, realw *Bnew,
																 realw *bz_semi_x11, realw *B, const ModelPara *d_mp, 
																 ModelGrid MG, int *yBzNew2old )
{
	// NOTE ix refers to y direction, iy represents x direction
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // 
	if( ix < d_mp->Any_New && iy< (d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx) )
	{
		__shared__ realw sh_dy[BLOCKSIZE3];
		__shared__ realw sh_bz[BLOCKSIZE3];
		__shared__ realw sh_mb[BLOCKSIZE3];

		volatile unsigned int n3=d_mp->Ay1 + d_mp->Ay2 + NPad;
		// copy from bz_semi_x01
		if( iy < d_mp->AnLx_New ) 
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + threadIdx.x];
				sh_bz[threadIdx.x] = bz_semi_x01[iy + threadIdx.x * d_mp->AnLx_New];
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];					// mb
			}
		}
		// copy from bz_old
		else if( iy >= d_mp->AnLx_New && iy < d_mp->AnLx_New + d_mp->Nx )
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + threadIdx.x];
				sh_bz[threadIdx.x] = Bold[iy- d_mp->AnLx_New + d_mp->ALx1 + d_mp->ALx2 
																	+ ( d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + threadIdx.x )*d_mp->L];	// REVIEW bz_old
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];					// mb
			}
		}	
		// copy from bz_semi_x11
		else
		{
			if( threadIdx.x < n3 ) 
			{
				sh_dy[threadIdx.x] = MG.dy[d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 - NPad + threadIdx.x];
				sh_bz[threadIdx.x] = bz_semi_x11[iy - (d_mp->Nx + d_mp->AnLx_New) + threadIdx.x * d_mp->AnRx_New];
				sh_mb[threadIdx.x] = B[iy*n3+threadIdx.x];					// mb
			}
		}
		__syncthreads( );

		// REVIEW why do we need this???
		int id = max( yBzNew2old[ix+d_mp->Ny+d_mp->Any_New] - 1 - (d_mp->Ny + d_mp->Ay1 + d_mp->Ay2-NPad), 0);
#if MAXDEBUG == 1
		if(ix == 0 && fabs(sh_mb[id]) < 0.01* fabs(sh_mb[id+1]) && fabs(sh_mb[id]) < 0.01* fabs(sh_mb[id+2])  )
			printf("iy=%d,id=%d,b=%e\n",iy,id,sh_mb[id]);
#endif		
		realw ma, mc;
		if( id < n3-1 )
		{
			ma = 1.0/3.0*(sh_mb[id+1]-sh_mb[id])/(0.5*sh_dy[id]+0.5*sh_dy[id+1]);
			mc =(sh_bz[id+1]-sh_bz[id])/(0.5*sh_dy[id]+0.5*sh_dy[id+1])
							- 1.0/3.0*(2.0*sh_mb[id]+sh_mb[id+1])*(0.5*sh_dy[id]+0.5*sh_dy[id+1]);						
		}
		else
		{
			ma = 1.0/3.0*(sh_mb[id]-sh_mb[id-1])/(0.5*sh_dy[id-1]+0.5*sh_dy[id]);
			mc =(sh_bz[id]-sh_bz[id-1])/(0.5*sh_dy[id-1]+0.5*sh_dy[id])
		          - 1.0/3.0*(2.0*sh_mb[id-1]+sh_mb[id])*(0.5*sh_dy[id-1]+0.5*sh_dy[id]);

			mc += 3.0*ma*(0.5*sh_dy[id-1]+0.5*sh_dy[id])*(0.5*sh_dy[id-1]+0.5*sh_dy[id]) 
						+ 2.0*sh_mb[id-1]*(0.5*sh_dy[id-1]+0.5*sh_dy[id]);
			ma = zerof;
		}
		  
		realw h = MG.Y_BzNew[ix+d_mp->Any_New+d_mp->Ny] - MG.Y_Bzold[id+(d_mp->Ny + d_mp->Ay1 + d_mp->Ay2-NPad)]; //REVIEW
		id = yBzNew2old[ix+d_mp->Ny+d_mp->Any_New]- 1 - (d_mp->Ny + d_mp->Ay1 + d_mp->Ay2-NPad);
		if( id >= 0 && id < n3-1 )
			Bnew[iy+(ix+d_mp->Any_New+d_mp->Ny)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] 
				= ma * h*h*h + sh_mb[id]*h*h + mc*h + sh_bz[id];
		else
			Bnew[iy+(ix+d_mp->Any_New+d_mp->Ny)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] 
				= ( sh_mb[max(id,0)]*h + mc )*h + sh_bz[max(id,0)];
#if MAXDEBUG == 2
		if(ix == 0 && fabs( Bnew[iy+(ix+d_mp->Any_New+d_mp->Ny)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] ) 
								< 0.01* fabs( Bnew[iy+(ix+1+d_mp->Any_New+d_mp->Ny)*(d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx)] ) )
			printf("iy=%d,id=%d,b=%e, b1=%e\n",iy,id,sh_mb[id],sh_mb[id+1]);
#endif					
	}
}

/* 
* ANCHOR interpolation copy the center regular grid area
*/
__global__ void Intep_BzIR2R_cpy_center( realw *Bold, realw *Bnew, const ModelPara *d_mp)
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; //
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; // 
	if( ix < d_mp->Nx && iy< d_mp->Ny )
	{
		Bnew[(iy+d_mp->Any_New)*(d_mp->AnLx_New+d_mp->Nx+d_mp->AnRx_New)+ix+d_mp->AnLx_New]		
		= Bold[(iy+d_mp->Ay1 + d_mp->Ay2)*(d_mp->L)+ix+d_mp->ALx1+d_mp->ALx2];		
	}
}		

/*
*	NOTE: This function is a CUDA implementation of the previous CPU code. It follows
*						exactly the same natural cubic spline interpolation/extrapolation algorithm
*						as of the CPU code, but with some modifications of logic:
*						---------------------------------------------------------------		
*			      |														Back															|		
*						|																															|
*						|							---------------------------------								|	
*						| 						|																|								|
*						|		Left			|						Center							|		right				|	
*						|							|				Regular grid						|								|
*						|							|																|								|
*						|							|																|								|
*						|							---------------------------------								|
*						|																															|
*						|													Front																|
*						|																															|
*						---------------------------------------------------------------
*
*		The center regular grid region needs no interpolation. We first do x direction interpolation
*		For the Left and right region (which could be concurrent); and then do y direction interpolation
*		for the front and back regions (the two could also be done simultanesously). But the y direction
*		interpolation should be done after that of the x direction, since it depends on the result of the
*		x direction interpolation in the overlaped corners.
*/
 int SplineGridConvIR2R( ModelPara *MP, ModelPara *d_mp, ModelGrid *MG, GridConv *GC, 
													UpContP *UPP, realw *Fold, cudaStream_t *stream )
{
	TRACE("SplineGridConvIR2R on GPU");
	unsigned int L1,M1;
	L1=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M1=2*MP->Any_New+MP->Ny;

	unsigned int n1=MP->ALx1 + MP->ALx2+NPad;
	unsigned int n2=MP->ARx1 + MP->ARx2+NPad;
	unsigned int n3=MP->Ay1 + MP->Ay2+NPad;
	
	cudaEvent_t Event[5];
	for(int ievt=0; ievt<5; ievt++)
		print_CUDA_error_if_any( cudaEventCreateWithFlags( &Event[ievt], cudaEventDisableTiming ), 510011 );
	//---------------------------------------------------------------------------------------------------
	//------------------------------------ central part copy --------------------------------------------
	//---------------------------------------------------------------------------------------------------
#if DEBUG == 2
	cudaEvent_t start, stop;
	start_timing_cuda(&start,&stop);	
#endif		
	// REVIEW This section is executed concurrently with the two sections below
	dim3 ThrdNum( BLOCKSIZE2x, BLOCKSIZE2y2 );
	dim3 gridcc( (MP->Nx + BLOCKSIZE2x - 1)/BLOCKSIZE2x , (MP->Ny + BLOCKSIZE2y2 - 1)/BLOCKSIZE2y2 );
	TRACE("Intep_BzIR2R_cpy_center");
	Intep_BzIR2R_cpy_center<<< gridcc, ThrdNum, 0, stream[2] >>>( Fold, UPP->BzAir, d_mp);
	GPU_ERROR_CHECKING("Intep_BzIR2R_cpy_center");

	// ANCHOR Event0
	print_CUDA_error_if_any( cudaEventRecord(Event[0], stream[2]), 510010 );

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "central part data copy");
	start_timing_cuda(&start,&stop);
#endif
	//--------------------------------------------------------------------------------------------------
	//------------------------------ x direction interpolation------------------------------------------
	//--------------------------------------------------------------------------------------------------

	//left side: filling the right hand side B matrix
	TRACE("Fill_B_0x");	
	if( n1 <= BLOCKSIZE3/2 )
		Fill_B_0x<<< MP->M, BLOCKSIZE3/2, 0, stream[0] >>>( Fold, (UPP->TDS0x).B, d_mp, *MG );
	else if( n1 > BLOCKSIZE3/2 && n1 <= BLOCKSIZE3 ) 
		Fill_B_0x<<< MP->M, BLOCKSIZE3, 0, stream[0] >>>( Fold, (UPP->TDS0x).B, d_mp, *MG );	
	else
	{
		exit_on_error("BLOCKSIZE3 (l222 in 'Cuda_device.h') is too small, it should be larger than AnLx1 + AnLx2 + NPad (AnRx1 + AnRx2 + NPad)");
	}	
	GPU_ERROR_CHECKING("Fill_B_0x");
	//right side: filling the right hand side B matrix
	TRACE("Fill_B_1x");
	if( n2 <= BLOCKSIZE3/2 )
		Fill_B_1x<<< MP->M, BLOCKSIZE3/2, 0, stream[1] >>>( Fold, (UPP->TDS1x).B, d_mp, *MG );
	else if( n2 > BLOCKSIZE3/2 && n2 <= BLOCKSIZE3 ) 
		Fill_B_1x<<< MP->M, BLOCKSIZE3, 0, stream[1] >>>( Fold, (UPP->TDS1x).B, d_mp, *MG );	
	else
	{
		exit_on_error("BLOCKSIZE3 (l222 in 'Cuda_device.h') is too small, it should be larger than AnLx1 + AnLx2 + NPad (AnRx1 + AnRx2 + NPad)");
	}	
	GPU_ERROR_CHECKING("Fill_B_1x");
#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "x direction right hand side matrix filling");
	start_timing_cuda(&start,&stop);	
#endif	
	//left side: solve the sparse matrix equations
#ifdef USE_SINGLE_PRECISION	
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot( (UPP->TDS0x).cuSPHandle, n1, MP->M, 
																(UPP->TDS0x).A.dl, (UPP->TDS0x).A.d, (UPP->TDS0x).A.du, (UPP->TDS0x).B,
																n1,  (void *)((UPP->TDS0x).buff) ), 42106 );
#else
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot( (UPP->TDS0x).cuSPHandle, n1, MP->M, 
																(UPP->TDS0x).A.dl, (UPP->TDS0x).A.d, (UPP->TDS0x).A.du, (UPP->TDS0x).B,
																n1,  (void *)((UPP->TDS0x).buff) ), 42107 );
#endif

	//right side: solve the sparse matrix equations
#ifdef USE_SINGLE_PRECISION	
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot( (UPP->TDS1x).cuSPHandle, n2, MP->M, 
																(UPP->TDS1x).A.dl, (UPP->TDS1x).A.d, (UPP->TDS1x).A.du, (UPP->TDS1x).B,
																n2,  (void *)((UPP->TDS1x).buff) ), 42206 );
#else
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot( (UPP->TDS1x).cuSPHandle, n2, MP->M, 
																(UPP->TDS1x).A.dl, (UPP->TDS1x).A.d, (UPP->TDS1x).A.du, (UPP->TDS1x).B,
																n2,  (void *)((UPP->TDS1x).buff) ), 42207 );
#endif
#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "x direction tridiagonal solver");
	start_timing_cuda(&start,&stop);	
#endif		
	//left side: interpolation
	dim3 grid0x( (MP->AnLx_New + 2*BLOCKSIZE3-1)/(2*BLOCKSIZE3) , MP->M );
	TRACE("Intep_BzIR2R_0x");
	Intep_BzIR2R_0x<<<grid0x,2*BLOCKSIZE3, 0, stream[0] >>>( Fold, (UPP->TDS0x).bz_semi0, UPP->BzAir,
																			(UPP->TDS0x).bz_semi1, (UPP->TDS0x).B, d_mp, *MG, GC->xBzNew2old );
	GPU_ERROR_CHECKING("Intep_BzIR2R_0x");

	// ANCHOR Event1
	print_CUDA_error_if_any( cudaEventRecord(Event[1], stream[0]), 510013 );	

	//right side: interpolation
	dim3 grid1x( (MP->AnRx_New + 2*BLOCKSIZE3-1)/(2*BLOCKSIZE3) , MP->M );
	TRACE("Intep_BzIR2R_1x");
	Intep_BzIR2R_1x<<<grid1x,2*BLOCKSIZE3, 0, stream[1] >>>( Fold, (UPP->TDS1x).bz_semi0, UPP->BzAir,
																			(UPP->TDS1x).bz_semi1, (UPP->TDS1x).B, d_mp, *MG, GC->xBzNew2old );
	GPU_ERROR_CHECKING("Intep_BzIR2R_1x");

	// ANCHOR Event2
	print_CUDA_error_if_any( cudaEventRecord(Event[2], stream[1]), 510014 );

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "x direction interpolation");
	start_timing_cuda(&start,&stop);	
#endif	

	// ANCHOR synchronisation
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[1], Event[1], 0), 510015 );
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[0], Event[2], 0), 510016 );
	//---------------------------------------------------------------------------------------------------
	//----------------------------- y direction interpolation--------------------------------------------
	//---------------------------------------------------------------------------------------------------
	// REVIEW this section must be executed after completing the above sections, there must be a syncronize
	//				mechanism if use the concurrent kernel execrution
	//Front: filling the right hand side B matrix
	TRACE("Fill_B_0y");
	if( n3 <= BLOCKSIZE3/2 )
		Fill_B_0y<<< L1, BLOCKSIZE3/2, 0, stream[0] >>>( Fold, (UPP->TDS0x).bz_semi0, (UPP->TDS1x).bz_semi0, (UPP->TDS0y).B, d_mp, *MG );
	else if( n3 > BLOCKSIZE3/2 && n3 <= BLOCKSIZE3 ) 
		Fill_B_0y<<< L1, BLOCKSIZE3, 0, stream[0]  >>>( Fold, (UPP->TDS0x).bz_semi0, (UPP->TDS1x).bz_semi0, (UPP->TDS0y).B, d_mp, *MG );	
	else
	{
		exit_on_error("BLOCKSIZE3 (l222 in 'Cuda_device.h') is too small, it should be larger than AnLx1 + AnLx2 +2 (AnRx1 + AnRx2 +2)");
	}	
	GPU_ERROR_CHECKING("Fill_B_0y");

	//Back: filling the right hand side B matrix
	TRACE("Fill_B_1y");
	if( n3 <= BLOCKSIZE3/2 )
		Fill_B_1y<<< L1, BLOCKSIZE3/2, 0, stream[1]  >>>( Fold, (UPP->TDS0x).bz_semi1, (UPP->TDS1x).bz_semi1, (UPP->TDS1y).B, d_mp, *MG );
	else if( n3 > BLOCKSIZE3/2 && n3 <= BLOCKSIZE3 ) 
		Fill_B_1y<<< L1, BLOCKSIZE3, 0, stream[1]  >>>( Fold, (UPP->TDS0x).bz_semi1, (UPP->TDS1x).bz_semi1, (UPP->TDS1y).B, d_mp, *MG );	
	else
	{
		exit_on_error("BLOCKSIZE3 (l222 in 'Cuda_device.h') is too small, it should be larger than AnLx1 + AnLx2 +2 (AnRx1 + AnRx2 +2)");
	}	
	GPU_ERROR_CHECKING("Fill_B_1y"); 
#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "y direction right hand side matrix filling");
	start_timing_cuda(&start,&stop);	
#endif	
	//Front side: solve the sparse matrix equations
#ifdef USE_SINGLE_PRECISION	
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot( (UPP->TDS0y).cuSPHandle, n3, L1, 
															(UPP->TDS0y).A.dl, (UPP->TDS0y).A.d, (UPP->TDS0y).A.du, (UPP->TDS0y).B,
															n3,  (void *)((UPP->TDS0y).buff) ), 42306 );
#else
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot( (UPP->TDS0y).cuSPHandle, n3, L1, 
															(UPP->TDS0y).A.dl, (UPP->TDS0y).A.d, (UPP->TDS0y).A.du, (UPP->TDS0y).B,
															n3,  (void *)((UPP->TDS0y).buff) ), 42307 );
#endif
	//Back side: solve the sparse matrix equations
#ifdef USE_SINGLE_PRECISION	
	print_cuSparse_error_if_any( cusparseSgtsv2_nopivot( (UPP->TDS1y).cuSPHandle, n3, L1, 
														(UPP->TDS1y).A.dl, (UPP->TDS1y).A.d, (UPP->TDS1y).A.du, (UPP->TDS1y).B,
														n3,  (void *)((UPP->TDS1y).buff) ), 42406 );
#else
	print_cuSparse_error_if_any( cusparseDgtsv2_nopivot( (UPP->TDS1y).cuSPHandle, n3, L1, 
														(UPP->TDS1y).A.dl, (UPP->TDS1y).A.d, (UPP->TDS1y).A.du, (UPP->TDS1y).B,
														n3,  (void *)((UPP->TDS1y).buff) ), 42407 );
#endif
#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "y direction tridiagonal solver");
	start_timing_cuda(&start,&stop);	
#endif
	//Front size: interpolation
	dim3 grid0y( (MP->Any_New + 2*BLOCKSIZE3-1)/(2*BLOCKSIZE3) , L1 );
	TRACE("Intep_BzIR2R_0y");
	Intep_BzIR2R_0y<<<grid0y,2*BLOCKSIZE3, 0, stream[0] >>>( Fold, (UPP->TDS0x).bz_semi0, UPP->BzAir,
																			(UPP->TDS1x).bz_semi0, (UPP->TDS0y).B, d_mp, *MG, GC->yBzNew2old );
	GPU_ERROR_CHECKING("Intep_BzIR2R_0y");

	// ANCHOR Event1
	print_CUDA_error_if_any( cudaEventRecord(Event[3], stream[0]), 510017 );

	//Back size: interpolation
	TRACE("Intep_BzIR2R_1y");
	Intep_BzIR2R_1y<<<grid0y,2*BLOCKSIZE3, 0, stream[1] >>>( Fold, (UPP->TDS0x).bz_semi1, UPP->BzAir,
																			(UPP->TDS1x).bz_semi1, (UPP->TDS1y).B, d_mp, *MG, GC->yBzNew2old );
	GPU_ERROR_CHECKING("Intep_BzIR2R_1y");

	// ANCHOR Event1
	print_CUDA_error_if_any( cudaEventRecord(Event[4], stream[1]), 510018 );

#if DEBUG == 2
	stop_timing_cuda(&start,&stop, "y direction interpolation");
#endif

	// ANCHOR synchronisation
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[0], Event[0], 0), 510019 );
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[0], Event[4], 0), 510020 );

	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[1], Event[0], 0), 510021 );
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[1], Event[3], 0), 510022 );

	// seems like we don't need this
	//print_CUDA_error_if_any( cudaStreamWaitEvent(stream[2], Event[3]), 510023 );
	//print_CUDA_error_if_any( cudaStreamWaitEvent(stream[2], Event[4]), 510024 );

for(int ievt=0; ievt<5; ievt++)
	print_CUDA_error_if_any( cudaEventDestroy( Event[ievt] ), 510012 );

	return 0;
}
