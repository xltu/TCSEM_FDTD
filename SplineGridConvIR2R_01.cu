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

*	XXX This GPU code use local 4 point natural cubic spline interpolation to replace
*			the global natural CSP in CPU algorithm, which should be faster, but possiblly
*			with a loss of interpolation accuracy 
*/
 
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "FTDT.h"
#include "Cuda_device.h"

// Naive Natural Cubic Spline interpolation for four points 
// Ref: https://www.math.ntnu.no/emner/TMA4215/2008h/cubicsplines.pdf
// used of keyward of __restrict__ could increase register pressure due to 'cached' loads
inline __device__ realw CubicSpline_1d4p( realw_const_p x, realw_const_p y, const realw t )
{
	const realw h1=x[2]-x[1];
	const realw v1=2.0*(x[2]-x[0]);
	const realw v2=2.0*(x[3]-x[1]);
	const realw u1=6.0*( (y[2]-y[1])/h1 - (y[1]-y[0])/(x[1]-x[0]) );
	const realw u2=6.0*( (y[3]-y[2])/(x[3]-x[2]) - (y[2]-y[1])/h1 );
	const realw z1= (v2*u1 - h1*u2)/(v1*v2 - h1*h1);
	const realw z2= (v1*u2 - h1*u1)/(v1*v2 - h1*h1);
	if(t < x[0])
		//return ( (y[1]-y[0])/(x[1]-x[0]) - z1*( x[1]-x[0])/6.0 ) * (t-x[0]) + x[0];
		return 0.0;
		
	else if ( t >= x[0] && t < x[1] )
		return z1/(6.0*(x[1]-x[0]))*MyPow(t-x[0],3) + (y[1]/(x[1]-x[0]) - z1/( 6.0 * (x[1]-x[0]) ))*(t-x[0]) + y[0]/(x[1]-x[0])*(x[1]-t); 
		
	else if ( t >= x[1] && t < x[2] )
		return z2/(6.0*h1)*MyPow(t-x[1],3) + z1/(6.0*h1)*MyPow(x[2]-t,3) + (y[2]/h1 - z2/6.0 * h1)*(t-x[1]) + (y[1]/h1 - z1/6.0 * h1)*(x[2]-t);
		
	else if ( t >= x[3] && t < x[3] )
		return z2/(6.0*(x[3]-x[2]))*MyPow(x[3]-t,3) + y[3]/(x[3]-x[2])*(t-x[2]) + (y[2]/(x[3]-x[2]) - z2/6.0 * (x[3]-x[2]))*(x[3]-t);
		
	else
		//return ( (y[3]-y[2])/(x[3]-x[2]) - z2*( x[3]-x[2])/6.0 ) * (t - x[3]) + x[3];	
		return 0.0;
}

__global__ void CubicSpline_BzIR2R( realw *Bold, realw *Bnew, const ModelPara *d_mp, 
																		ModelGrid MG, GridConv GC)
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	volatile unsigned int L1=d_mp->AnLx_New + d_mp->AnRx_New + d_mp->Nx;
  volatile unsigned int M1=2*d_mp->Any_New + d_mp->Ny;
  if(ix < L1 && iy < M1 )
  {
		const int II = min( max( GC.xBzNew2old[ix], d_mp->L-2 ), 2);
		const int JJ = min( max( GC.yBzNew2old[iy], d_mp->M-2 ), 2);
		
		realw Y[4],bz[4];
		// x dir interpolation
		#pragma unroll
		for(int k=0; k<4; k++)
		{
			Y[k] = get_global_cr( &MG.Y_BzNew[JJ-2 + k] );
			bz[k] = CubicSpline_1d4p( &MG.X_Bzold[II-2], &Bold[(JJ-2 + k) * d_mp->L + II-2],  MG.X_BzNew[ix]);
		}	
		
		Bnew[iy*L1+ix] = CubicSpline_1d4p( Y, bz,  MG.Y_BzNew[iy]);
	}	
}	

 int SplineGridConvIR2R(ModelPara *MP, ModelPara *d_mp, ModelGrid *MG, GridConv *GC, realw *Fold, realw *Fnew)
{
	TRACE("SplineGridConvIR2R on GPU");
	int L1,M1;
	L1=MP->AnLx_New+MP->AnRx_New+MP->Nx;
  M1=2*MP->Any_New+MP->Ny;
	
	dim3 threads(BLOCKSIZE2x,BLOCKSIZE2y2);
	dim3 grid( (L1 + BLOCKSIZE2x-1)/BLOCKSIZE2x , (M1 + BLOCKSIZE2y2-1)/BLOCKSIZE2y2 );
	
	CubicSpline_BzIR2R<<<grid,threads>>>( Fold, Fnew, d_mp, *MG, *GC);
	
	GPU_ERROR_CHECKING("SplineGridConvIR2R on GPU");

	return 0;
}
