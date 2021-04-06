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
 /* performance statistics for the kernels */
 // limiting the max number of Rx per Tx to ~800
 // Ex: 64 registers
 // Ey: 74 registers
 // Ez: 88 registers
 // Bx: 78 registers
 // By: 78 registers
 // Bz: 66 registers
 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>
//#include<omp.h>
#include"FTDT.h"
#include"Cuda_device.h"
//#include"prepare_constants_cuda.h"

// Naive Natural Cubic Spline interpolation for four points 
// Ref: https://www.math.ntnu.no/emner/TMA4215/2008h/cubicsplines.pdf
// used of keyward of __restrict__ could increase register pressure due to 'cached' loads
inline __device__ realw CubicSpline_1d4p( realw *x, realw *y, const realw t )
{
	const realw h1=x[2]-x[1];
	const realw v1=2.0*(x[2]-x[0]);
	const realw v2=2.0*(x[3]-x[1]);
	const realw u1=6.0*( (y[2]-y[1])/h1 - (y[1]-y[0])/(x[1]-x[0]) );
	const realw u2=6.0*( (y[3]-y[2])/(x[3]-x[2]) - (y[2]-y[1])/h1 );
	const realw z1= (v2*u1 - h1*u2)/(v1*v2 - h1*h1);
	const realw z2= (v1*u2 - h1*u1)/(v1*v2 - h1*h1);
	
	return z2/(6.0*h1)*MyPow(t-x[1],3) + z1/(6.0*h1)*MyPow(x[2]-t,3) + (y[2]/h1 - z2/6.0 * h1)*(t-x[1]) + (y[1]/h1 - z1/6.0 * h1)*(x[2]-t);
}

// nr and base are passed by value, in constant mem
__global__ void ExSpatialInterp( realw *Ex, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
#if MAXDEBUG == 1
	printf("Rx #%5d",RxLst[ix]);
#endif			
		realw t1[4],Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		int iskep1,iskep2,iskep3;
		// interpolation along x direction
		// X from (MG.X_Bzold)[iskep1] to (MG.X_Bzold)[iskep1+3]
		iskep1=(xyzRx.ix)[ir]-2;
		
		// interpolation along y direction
		if( ( get_global_cr(&(MG.Y_Bzold)[(xyzRx.iy)[ir]-1]) + 0.5*get_global_cr(&(MG.dy)[(xyzRx.iy)[ir]-1]) ) > get_global_cr(&(xyzRx.y)[ir]) )
			iskep2=(xyzRx.iy)[ir]-2;
		else
			iskep2=(xyzRx.iy)[ir]-1;
			
		// z direction interpolation
		if( ( get_global_cr(&(MG.Z)[(xyzRx.iz)[ir]-1]) + 0.5*get_global_cr(&(MG.dz)[(xyzRx.iz)[ir]-1]) ) > get_global_cr(&(xyzRx.z)[ir]) )
			iskep3=(xyzRx.iz)[ir]-2;
		else
			iskep3=(xyzRx.iz)[ir]-1;
		if( iskep3<0 )
			iskep3=0;	
			
		//set t1 as Y coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t1[ij]=(MG.Y_Bzold)[iskep2+ij]-0.5*(MG.dy)[iskep2+ij];	
			
		volatile unsigned int ij,indx;
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation	
				indx = ( (ik+iskep3) * (mp->M+1) + ij + iskep2 ) * mp->L + iskep1;
				Y1[ij]=CubicSpline_1d4p( &((MG.X_Bzold)[iskep1]), &Ex[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( t1, Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
			
		//set t1 as Z coordinates
		#pragma unroll
		for(ij=0;ij<4;ij++)
			t1[ij]=(MG.Z)[iskep3+ij]-0.5*(MG.dz)[iskep3+ij];	
		
		ef0[Base+ix]=CubicSpline_1d4p( t1, Y2, (xyzRx.z)[ir] );
		
	}
}	

// nr and base are passed by value, in constant mem
__global__ void EySpatialInterp( realw *Ey, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		
		realw t1[4],Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		int iskep1,iskep2,iskep3;
		// interpolation along x direction
		if( ( get_global_cr(&(MG.X_Bzold)[(xyzRx.ix)[ir]-1]) + 0.5*get_global_cr(&(MG.dx)[(xyzRx.ix)[ir]-1]) ) > get_global_cr(&(xyzRx.x)[ir]) )
			iskep1=(xyzRx.ix)[ir]-2;
		else
			iskep1=(xyzRx.ix)[ir]-1;
				
		// interpolation along y direction
		iskep2=(xyzRx.iy)[ir]-2;
			
		// z direction interpolation
		if( ( get_global_cr(&(MG.Z)[(xyzRx.iz)[ir]-1]) + 0.5*get_global_cr(&(MG.dz)[(xyzRx.iz)[ir]-1]) ) > get_global_cr(&(xyzRx.z)[ir]) )
			iskep3=(xyzRx.iz)[ir]-2;
		else
			iskep3=(xyzRx.iz)[ir]-1;
		if( iskep3<0 )
			iskep3=0;	
			
		//set t1 as X coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t1[ij]=(MG.X_Bzold)[iskep1+ij]-0.5*(MG.dx)[iskep1+ij];	
			
		volatile unsigned int ij, indx;
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation		
				indx= ( (ik+iskep3) * (mp->M) + ij + iskep2 ) * (mp->L + 1) + iskep1;
				Y1[ij]=CubicSpline_1d4p( t1, &Ey[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( &(MG.Y_Bzold)[iskep2], Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
			
		//set t1 as Z coordinates
		#pragma unroll
		for(ij=0;ij<4;ij++)
			t1[ij]=(MG.Z)[iskep3+ij]-0.5*(MG.dz)[iskep3+ij];	
		
		ef0[Base+ix]=CubicSpline_1d4p( t1, Y2, (xyzRx.z)[ir] );
	}
}	

// nr and base are passed by value, in constant mem
__global__ void EzSpatialInterp( realw *Ez, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{	
		realw t1[4],t2[4],Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		volatile int iskep1,iskep2,iskep3;
		// interpolation along x direction
		if( ( get_global_cr(&(MG.X_Bzold)[(xyzRx.ix)[ir]-1]) + 0.5*get_global_cr(&(MG.dx)[(xyzRx.ix)[ir]-1]) ) > get_global_cr(&(xyzRx.x)[ir]) )
			iskep1=(xyzRx.ix)[ir]-2;
		else
			iskep1=(xyzRx.ix)[ir]-1;
				
		// interpolation along y direction
		if( ( get_global_cr(&(MG.Y_Bzold)[(xyzRx.iy)[ir]-1]) + 0.5*get_global_cr(&(MG.dy)[(xyzRx.iy)[ir]-1]) ) > get_global_cr(&(xyzRx.y)[ir]) )
			iskep2=(xyzRx.iy)[ir]-2;
		else
			iskep2=(xyzRx.iy)[ir]-1;
			
		// z direction interpolation
		iskep3=(xyzRx.iz)[ir]-2;
		if( iskep3<0 )
			iskep3=0;	
			
		//set t1 as X coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t1[ij]=(MG.X_Bzold)[iskep1+ij]-0.5*(MG.dx)[iskep1+ij];	
			
		//set t2 as Y coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t2[ij]=(MG.Y_Bzold)[iskep2+ij]-0.5*(MG.dy)[iskep2+ij];	
			
		volatile unsigned int indx;
		volatile unsigned int ij;	
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation	
				indx=( (ik+iskep3) * (mp->M+1) + ij + iskep2 ) * (mp->L+1) + iskep1;
				Y1[ij]=CubicSpline_1d4p( t1, &Ez[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( t2, Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
		
		ef0[Base+ix]=CubicSpline_1d4p( &(MG.Z)[iskep3], Y2, (xyzRx.z)[ir] );
	}
}	

// nr and base are passed by value, in constant mem
__global__ void BxSpatialInterp( realw *Bx, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		
		realw t1[4],Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		int iskep1,iskep2,iskep3;
		// interpolation along x direction
		if( ( get_global_cr(&(MG.X_Bzold)[(xyzRx.ix)[ir]-1]) + 0.5*get_global_cr(&(MG.dx)[(xyzRx.ix)[ir]-1]) ) > get_global_cr(&(xyzRx.x)[ir]) )
			iskep1=(xyzRx.ix)[ir]-2;
		else
			iskep1=(xyzRx.ix)[ir]-1;
				
		// interpolation along y direction
		iskep2=(xyzRx.iy)[ir]-2;
			
		// z direction interpolation
		iskep3=(xyzRx.iz)[ir]-2;
		if( iskep3<0 )
			iskep3=0;	
			
		//set t1 as X coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t1[ij]=(MG.X_Bzold)[iskep1+ij]-0.5*(MG.dx)[iskep1+ij];	
			
		volatile unsigned int ij, indx;
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation
				indx=( (ik+iskep3) * (mp->M) + ij + iskep2 ) * (mp->L+1) + iskep1;
				Y1[ij]=CubicSpline_1d4p( t1, &Bx[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( &(MG.Y_Bzold)[iskep2], Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
		
		ef0[Base+ix]=CubicSpline_1d4p( &(MG.Z)[iskep3], Y2, (xyzRx.z)[ir] );
	}
}	

// nr and base are passed by value, in constant mem
__global__ void BySpatialInterp( realw *By, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		
		realw t1[4],Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		int iskep1,iskep2,iskep3;
		// interpolation along x direction
		iskep1=(xyzRx.ix)[ir]-2;
				
		// interpolation along y direction
		if( ( get_global_cr(&(MG.Y_Bzold)[(xyzRx.iy)[ir]-1]) + 0.5*get_global_cr(&(MG.dy)[(xyzRx.iy)[ir]-1]) ) > get_global_cr(&(xyzRx.y)[ir]) )
			iskep2=(xyzRx.iy)[ir]-2;
		else
			iskep2=(xyzRx.iy)[ir]-1;	
			
		// z direction interpolation
		iskep3=(xyzRx.iz)[ir]-2;
		if( iskep3<0 )
			iskep3=0;	
			
		//set t1 as Y coordinates
		#pragma unroll
		for(int ij=0;ij<4;ij++)
			t1[ij]=(MG.Y_Bzold)[iskep2+ij]-0.5*(MG.dy)[iskep2+ij];	
			
		volatile unsigned int ij, indx;
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation
				indx=( (ik+iskep3) * (mp->M+1) + ij + iskep2 ) * mp->L + iskep1;
				Y1[ij]=CubicSpline_1d4p( &(MG.X_Bzold)[iskep1], &By[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( t1, Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
		
		ef0[Base+ix]=CubicSpline_1d4p( &(MG.Z)[iskep3], Y2, (xyzRx.z)[ir] );
	}
}
// nr and base are passed by value, in constant mem
__global__ void BzSpatialInterp( realw *Bz, const ModelPara *mp, const ModelGrid MG, 
																RxPos xyzRx, int *RxLst, realw *ef0, int nr, int Base) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		
		realw Y1[4],Y2[4];
		// rx position in the RxPos
		const int ir=RxLst[ix]-1;	
		int iskep1,iskep2,iskep3;
		// interpolation along x direction
		iskep1=(xyzRx.ix)[ir]-2;
				
		// interpolation along y direction
		iskep2=(xyzRx.iy)[ir]-2;
		
		// z direction interpolation
		if( ( get_global_cr(&(MG.Z)[(xyzRx.iz)[ir]-1]) + 0.5*get_global_cr(&(MG.dz)[(xyzRx.iz)[ir]-1]) ) > get_global_cr(&(xyzRx.z)[ir]) )
			iskep3=(xyzRx.iz)[ir]-2;
		else
			iskep3=(xyzRx.iz)[ir]-1;
		if( iskep3<0 )
			iskep3=0;	
			
		volatile unsigned int ij, indx;
		#pragma unroll
		for(int ik=0; ik<4; ik++)
		{
			for(ij=0; ij<4; ij++)
			{
				// x direction interpolation
				indx=	( (ik+iskep3) * (mp->M) + ij + iskep2 ) * mp->L + iskep1;
				Y1[ij]=CubicSpline_1d4p( &(MG.X_Bzold)[iskep1], &Bz[indx], get_global_cr(&(xyzRx.x)[ir]) );
			}
			// y direction interpolation
			Y2[ik]=CubicSpline_1d4p( &(MG.Y_Bzold)[iskep2], Y1, get_global_cr(&(xyzRx.y)[ir]) );
		}		
		
		//set Y1 as Z coordinates
		#pragma unroll
		for(ij=0;ij<4;ij++)
			Y1[ij]=(MG.Z)[iskep3+ij]-0.5*(MG.dz)[iskep3+ij];	
		
		ef0[Base+ix]=CubicSpline_1d4p( Y1, Y2, (xyzRx.z)[ir] );
	}
}
// Electric field time interp
__global__ void EFTimeInterp( realw *ef0, realw *efs, int nr, int Nt, int ne_efs, 
															int nefcmp, int NComp, realw l0, realw l1) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		#pragma unroll
		for(int icomp=0; icomp<nefcmp; icomp++)
		{
			efs[ix*NComp*Nt + icomp*Nt + ne_efs]=l0*ef0[icomp*nr+ix]+l1*ef0[NComp*nr+icomp*nr+ix];
		}
	}
}

// magnetic induction field time interp
__global__ void BFTimeInterp( realw *ef0, realw *efs, int nr, int Nt, int ne_efs, 
															int nefcmp, int NComp, realw l0, realw l1) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	if(ix<nr)
	{
		#pragma unroll
		for(int icomp=nefcmp; icomp<NComp; icomp++)
		{
			efs[ix*NComp*Nt + icomp*Nt + ne_efs]=l0*ef0[icomp*nr+ix]+l1*ef0[NComp*nr+icomp*nr+ix];
		}
	}
}


// interpolate the E field to the receiver position
int EFXInterp(DArrays *DPtrs, ModelGrid *d_MG, ModelPara *d_mp, 
							RxTime tRx, int nr, RxPos *d_xyzRx, int *RxLst,
							realw *ef0, int Base, cudaStream_t stream)
{
#if MAXDEBUG == 1
	printf("Total number of Rx is %d\n",nr);
#endif	
	int im; 
	if( tRx.isComp[0] )
	{
		//Ex component
		ExSpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->Ex, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base); 
	}
	//Ey
	if( tRx.isComp[1] )
	{
		im=tRx.isComp[0] * nr;
		EySpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->Ey, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base+im); 
	}
	//Ez
	if( tRx.isComp[2] )
	{
		im=(tRx.isComp[0]+tRx.isComp[1]) * nr;
		EzSpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->Ez, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base+im); 
	}
	return 0;
}

// interpolate the B field to the receiver position
int BFXInterp(DArrays *DPtrs, ModelGrid *d_MG, ModelPara *d_mp, 
							RxTime tRx, int nr, RxPos *d_xyzRx, int *RxLst,
							realw *ef0, int Base, cudaStream_t stream)
{	
	int im; 
	// Bx
	if( tRx.isComp[3] )
	{
		im=tRx.nefcmp*nr;
		BxSpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->Bx, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base+im); 
	}
	//By
	if( tRx.isComp[4] )
	{
		im=(tRx.nefcmp+tRx.isComp[3]) * nr;
		BySpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->By, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base+im); 
	}
	//Bz
	if( tRx.isComp[5] )
	{
		im=(tRx.nefcmp+tRx.isComp[3]+tRx.isComp[4]) * nr;
		BzSpatialInterp<<<(nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream >>>( DPtrs->Bz, d_mp, *d_MG, 
																*d_xyzRx, RxLst, ef0, nr, Base+im); 
	}
	return 0;
}

int EFInterp2Rx(int *ne_efs, int *nb_efs, RxTime tRx, TxPos xyzTx, RxPos *d_xyzRx, int *RxLst,
								ModelGrid *d_MG, ModelPara *d_mp, DArrays *DPtrs, realw t0, realw dt, 
								realw Tpof[2][3], realw *efs, realw *ef0, cudaStream_t *stream)
{
	realw l1;
	//-------------------------------------electric field----------------------------------------------------------------
	if( *ne_efs < tRx.Nt)
	{
		//printf("Sampling time=%fs, Pre. E ite time=%f, Cur. E ite time=%f, next E ite time=%f\n",tRx.Times[*ne_efs],Tpof[0][0],Tpof[0][1],t0);
		// next time steping 
		Tpof[0][2]=t0;
		// next sampling time is smaller than the previous ite time step, probably 
		// because the initial sampling time is too small or the sampling point
		// tag is not updated 
		if( tRx.Times[*ne_efs] <= Tpof[0][0] )
		{
			while( *ne_efs < tRx.Nt && tRx.Times[*ne_efs] <= Tpof[0][0] )
				(*ne_efs)++;	
		}
				
		// next sampling time is after the next time iteration, do nothing
		//if( tRx.Times[*ne_efs] > Tpof[0][2] ) 
		
		// next sampling time is in the middle of the current and next time iterations, 
		// specially interpolate the E field of the current time step to the receiver locations 
		if( tRx.Times[*ne_efs] <= Tpof[0][2] && tRx.Times[*ne_efs] > Tpof[0][1] )
		{
			// save to ef0[0][][]
			//EFXInterp(MG, EF, tRx, xyzTx, xyzRx, ef0[0]);		
			EFXInterp(DPtrs, d_MG, d_mp, tRx, xyzTx.nr, d_xyzRx, RxLst, ef0, 0, stream[0]);
		}
		// next sampling time is in the middle of the previous and current time iteration
		// specially interpolate the E field of the current time step to the receiver locations
		// temporarily interpolate the E field to the next sampling time
		// step the sampling time forward to make sure it's bigger than the current time step (tRx.Times[*nefs] <= Tpof[0][1]) 
		else if( tRx.Times[*ne_efs] <= Tpof[0][1]  && tRx.Times[*ne_efs] > Tpof[0][0] )  	
		{
			// save to ef0[1][][]
			//EFXInterp(MG, EF, tRx, xyzTx, xyzRx, ef0[1]);
			EFXInterp(DPtrs, d_MG, d_mp, tRx, xyzTx.nr, d_xyzRx, RxLst, ef0, tRx.NComp*xyzTx.nr, stream[0]);
			// temporial interpolation to the rx sampling time
			while(*ne_efs < tRx.Nt && tRx.Times[*ne_efs] <= Tpof[0][1] )
			{
				l1=(tRx.Times[*ne_efs]-Tpof[0][0])/(Tpof[0][1]-Tpof[0][0]);
				// E field temporial interpolation
				
				EFTimeInterp<<<(xyzTx.nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream[0] >>>( ef0, efs, 
																								xyzTx.nr, tRx.Nt, *ne_efs, tRx.nefcmp, tRx.NComp, 1.0-l1, l1);
				
				(*ne_efs)++;
			}			
		}	
		else
		;
		
		Tpof[0][0]=Tpof[0][1];
		Tpof[0][1]=Tpof[0][2];	
	}	
	//-------------------------------------------------------------------------------------------------------------------------
	
	
	//------------------------------------------------- B component------------------------------------------------------------
	if( *nb_efs < tRx.Nt && ( tRx.isComp[3] || tRx.isComp[4] || tRx.isComp[5] ) )
	{
		//printf("Sampling time=%fs, Pre. B ite time=%f, Cur. E ite time=%f, next B ite time=%f\n",tRx.Times[*ne_efs],Tpof[1][0],Tpof[1][1],t0+dt/2);
		// next time steping for B field
		Tpof[1][2]=t0 + 0.5*dt;
		// next sampling time is smaller than the previous ite time step, probably 
		// because the initial sampling time is too small or the sampling point
		// tag is not updated 
		if( tRx.Times[*nb_efs] <= Tpof[1][0] )
		{	
			while( *nb_efs < tRx.Nt && tRx.Times[*nb_efs] <= Tpof[1][0] )
				(*nb_efs)++;	
		}
		
		// next sampling time is in the middle of the current and next time iterations, 
		// specially interpolate the E field of the current time step to the receiver locations 
		if( tRx.Times[*nb_efs] <= Tpof[1][2] && tRx.Times[*nb_efs] > Tpof[1][1] )
		{
			// save to ef0[0][][]
			//BFXInterp(MG, BF, tRx, xyzTx, xyzRx, ef0[0]);		
			BFXInterp(DPtrs, d_MG, d_mp, tRx, xyzTx.nr, d_xyzRx, RxLst, ef0, 0, stream[1]);
		}
		// next sampling time is in the middle of the previous and current time iteration
		// specially interpolate the E field of the current time step to the receiver locations
		// temporarily interpolate the E field to the next sampling time
		// step the sampling time forward to make sure it's bigger than the current time step (tRx.Times[*nefs] <= Tpof[0][1]) 
		else if( tRx.Times[*nb_efs] <= Tpof[1][1]  && tRx.Times[*nb_efs] > Tpof[1][0] )  	
		{
			// save to ef0[1][][]
			//BFXInterp(MG, BF, tRx, xyzTx, xyzRx, ef0[1]);
			BFXInterp(DPtrs, d_MG, d_mp, tRx, xyzTx.nr, d_xyzRx, RxLst, ef0, tRx.NComp*xyzTx.nr, stream[1]);
							
			// temporial interpolation to the rx sampling time
			while( *nb_efs < tRx.Nt && tRx.Times[*nb_efs] <= Tpof[1][1] )
			{
				l1=(tRx.Times[*nb_efs]-Tpof[1][0])/(Tpof[1][1]-Tpof[1][0]);
				BFTimeInterp<<<(xyzTx.nr+BLOCKSIZE1-1)/BLOCKSIZE1,BLOCKSIZE1, 0, stream[1] >>>( ef0, efs, 
																								xyzTx.nr, tRx.Nt, *nb_efs, tRx.nefcmp, tRx.NComp, 1.0-l1, l1);
				
				(*nb_efs)++;
			}			
		}	
		else
		;
		
		Tpof[1][0]=Tpof[1][1];
		Tpof[1][1]=Tpof[1][2];	
	}
	//----------------------------------------------------------------------------------------------------------------------------------------------
	
	return 0;
}




