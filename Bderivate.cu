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
/* the founction for caculate time derivate of magnetic induction in every time step*/
// @XT Changed the padded mem to linear mem for EM field and conductivity
//
#include<stdio.h>
#include"FTDT.h"
#include"Cuda_device.h"
//#include"prepare_constants_cuda.h"

// calculate the grid size in x direction
inline __device__ realw Grid_dx( const ModelPara *d_mp, const int i )
{
	// note i here equals to i in the CPU code
	if( i >= d_mp->ALx2 + d_mp->ALx1 && i < d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 )
		return d_mp->dxmin;
	else if( i >= d_mp->ALx2 && i < d_mp->ALx2 + d_mp->ALx1 )
		return d_mp->dxmin *MyPow( d_mp->arg1 , d_mp->ALx2 + d_mp->ALx1 -i );
	else if( i < d_mp->ALx2 )
		return d_mp->dxmin * MyPow( d_mp->arg1 , d_mp->ALx1 ) * MyPow( d_mp->arg2 , d_mp->ALx2 -i );
	else if( i >= d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 && i < 	d_mp->Nx + d_mp->ARx1 + d_mp->ALx1 + d_mp->ALx2 )
		return d_mp->dxmin *MyPow( d_mp->arg1 , i - d_mp->Nx - d_mp->ALx1 - d_mp->ALx2 + 1 );
	else
		return d_mp->dxmin * MyPow( d_mp->arg1 , d_mp->ARx1 ) * MyPow( d_mp->arg2 , i - d_mp->Nx - d_mp->ARx1 - d_mp->ALx1 - d_mp->ALx2 + 1 );	
}

// calculate the grid size in y direction
inline __device__ realw Grid_dy( const ModelPara *d_mp, const int j )
{
	// note j here equals to j in the CPU code
	if( j >= d_mp->Ay2 + d_mp->Ay1 && j < d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 )
		return d_mp->dymin;
	else if( j >= d_mp->Ay2 && j < d_mp->Ay2 + d_mp->Ay1 )
		return d_mp->dymin *MyPow( d_mp->arg1 , d_mp->Ay2 + d_mp->Ay1 -j );
	else if( j < d_mp->Ay2 )
		return d_mp->dymin * MyPow( d_mp->arg1 , d_mp->Ay1 ) * MyPow( d_mp->arg2 , d_mp->Ay2 -j );
	else if( j >= d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 && j < 	d_mp->Ny + 2 * d_mp->Ay1 + d_mp->Ay2 )
		return d_mp->dymin *MyPow( d_mp->arg1 , j - d_mp->Ny - d_mp->Ay1 - d_mp->Ay2 + 1 );
	else
		return d_mp->dymin * MyPow( d_mp->arg1 , d_mp->Ay1 ) * MyPow( d_mp->arg2 , j - d_mp->Ny - 2*d_mp->Ay1 - d_mp->Ay2 + 1 );	
}

// calculate the grid size in z direction
inline __device__ realw Grid_dz( const ModelPara *d_mp, const int i )
{
	// note i here equals to k in the CPU code
	if( i < d_mp->Nz )
		return d_mp->dzmin;
	else if( i >= d_mp->Nz && i < d_mp->Nz + d_mp->Az1 )
		return d_mp->dzmin *MyPow( d_mp->arg1 , i- d_mp->Nz + 1 );
	else
		return d_mp->dzmin * MyPow( d_mp->arg1 , d_mp->Az1 ) * MyPow( d_mp->arg2 , i - d_mp->Nz - d_mp->Az1 + 1 );
	
}

// dt is passed by value, in constant mem
__global__ void UpdateBx( realw *Bx, realw *Ey, realw *Ez,
													const ModelPara *d_mp, const realw dt ) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-2]; iy in [0,M-1]; iz in [0,N-1]
	// ix=i-1, iy=j, iz=k; transform to index in cpu code
	if( ix < L-1 && iy < M )
	{
		//const int N=d_mp->N;
		volatile unsigned int k;
		// smem of size 17*16*8=2176B
		__shared__ realw sh_ez[BLOCKSIZE2y2 + 1][BLOCKSIZE2x];
		__shared__ realw dz;
		// determine dy[j], keep constant for different z
		const realw dy = Grid_dy(d_mp,iy);
		//const realw dy = d_dy[iy];
		// reuse ey
		field ey2 = Make_field( Ey[iy*(L+1)+ix+1], zerof );//Ey[0][j][i]		
		
		for(k=0;k<d_mp->N;k++)
		{
			//cpy Ez to sh_ez
			volatile unsigned int indx=( k * (M+1) + iy )*(L+1)+ix+1;
			sh_ez[threadIdx.y][threadIdx.x]= Ez[indx];			// Ez[k][j][i], only read once, no need to load to cache	
			if(threadIdx.y == BLOCKSIZE2y2-1 || iy == M-1)
			{	
				indx=( k * (M+1) + iy + 1)*(L+1)+ix+1;
				sh_ez[threadIdx.y+1][threadIdx.x]= get_global_cr( &Ez[indx] );			// Ez[k][j+1][i], need by another block, load to cache								
			}
			if(threadIdx.x==0 && threadIdx.y==0)
				dz=Grid_dz(d_mp,k);
				
			__syncthreads( );
			
			//read in Ey
			ey2.y=ey2.x;
			ey2.x= Ey[ ( (k+1) * M + iy ) * (L+1) + ix+1];										//Ey[k+1][j][i], read only once
			
			// Bx[k][j][i]
			//row = (realw *)((char *)Bx.ptr + ( k * M + iy ) * Bx.pitch);	 
			//row[ix+1] += ( diff(ey2)/Grid_dz(d_mp,k) - 
			//								(sh_ez[threadIdx.y+1][threadIdx.x] - sh_ez[threadIdx.y][threadIdx.x])/dy )*dt;
			Bx[( k * M + iy ) * (L+1) + ix+1] += ( diff(ey2)/dz - 
											(sh_ez[threadIdx.y+1][threadIdx.x] - sh_ez[threadIdx.y][threadIdx.x])/dy )*dt;
			
		}
#if MAXDEBUG == 1
		if(isnan(Bx[iy*(L+1) + ix+1]))
		{
			printf("error in Bx[0][%d][%d]\n",iy,ix);
			printf("dz=%f\n",dz);
			printf("ez:%e - %e\n",Ez[(iy + 1) * (L + 1) + ix+1], Ez[(iy) * (L + 1) + ix+1]);
			printf("ey:%e - %e\n",Ey[(M+ iy) * (L+1) + ix+1], Ey[(iy) * (L+1) + ix+1]);
		}	
#endif		
		
	}
}

__global__ void UpdateBy( realw *By, realw *Ex, realw *Ez,
													const ModelPara *d_mp, const realw dt ) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-1]; iy in [0,M-2]; iz in [0,N-1]
	// ix=i, iy=j-1, iz=k; transform to index in cpu code
	if( ix < L && iy < M-1 )
	{
		//const int N=d_mp->N;
		volatile unsigned int k;
		// smem of size 17*16*8=2176B
		__shared__ realw sh_ez[BLOCKSIZE2y2][BLOCKSIZE2x+1];
		__shared__ realw dz;
		// determine dx[i], keep constant for different z
		const realw dx = Grid_dx(d_mp,ix);
		
		// reuse ex	
		field ex2 = Make_field( Ex[(iy + 1) * L + ix], zerof );//Ex[0][j][i]		
		
		for(k=0;k<d_mp->N;k++)
		{
			//cpy Ez to sh_ez
			//row = (realw *)((char *)Ez.ptr + ( k * (M+1) + iy + 1) * Ez.pitch);
			volatile unsigned int indx=( k * (M+1) + iy + 1) * (L + 1) + ix;
			sh_ez[threadIdx.y][threadIdx.x]= Ez[indx];			// Ez[k][j][i], only read once, no need to load to cache	
			if(threadIdx.x == BLOCKSIZE2x-1 || ix == L-1)
				sh_ez[threadIdx.y][threadIdx.x+1]= get_global_cr( &Ez[indx+1] );			// Ez[k][j][i+1], need by another block, load to cache								
			
			if(threadIdx.x==0 && threadIdx.y==0)
				dz=Grid_dz(d_mp,k);
				
			__syncthreads( );
			
			//read in Ex
			ex2.y=ex2.x;
			//row = (realw *)((char *)Ex.ptr + ( (k+1) * (M+1) + iy + 1) * Ex.pitch);	
			ex2.x= Ex[( (k+1) * (M+1) + iy + 1) * L + ix];										//Ey[k+1][j][i], read only once
			
			// By[k][j][i]
			//row = (realw *)((char *)By.ptr + ( k * (M+1) + iy + 1) * By.pitch);	 
			By[ ( k * (M+1) + iy + 1)*L + ix] += ( (sh_ez[threadIdx.y][threadIdx.x+1] - sh_ez[threadIdx.y][threadIdx.x])/dx - 
											 diff(ex2)/dz )*dt;
			
		}
#if MAXDEBUG == 1
		if(isnan(By[ (iy + 1)*L + ix]))
		{
			printf("error in By[0][%d][%d]\n",iy,ix);
			printf("dz=%f\n",dz);
			printf("ez:%e - %e\n",Ez[(iy + 1) * (L + 1) + ix], Ez[(iy + 1) * (L + 1) + ix+1]);
			printf("ex:%e - %e\n",Ex[(M+1+ iy + 1) * L + ix], Ex[(iy + 1) * L + ix]);
		}	
#endif
	}
}

__global__ void UpdateBz( realw *Bz, realw *Bx, realw *By, const ModelPara *d_mp) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-1]; iy in [0,M-1]; iz in [0,N-1]
	// ix=i, iy=j, iz=k; transform to index in cpu code
	if( ix < L && iy < M )
	{
		//const int N=d_mp->N;
		//volatile unsigned int k;
		realw bz0=0.0;	//reuse bz
		//realw *row;
		// smem of size 17*16*2*8=4353B
		__shared__ realw sh_bx[BLOCKSIZE2y1][BLOCKSIZE2x+1];
		__shared__ realw sh_by[BLOCKSIZE2y1+1][BLOCKSIZE2x];
		//__shared__ realw dx;
		//__shared__ realw dy;
		// dx[i] and dy[j] keep constant for different z
		const realw dx = Grid_dx(d_mp,ix);
		const realw dy = Grid_dy(d_mp,iy);
		
		// Bz is calculated from bottom up		
		for(volatile int k=d_mp->N-1;k>=0;k--)
		{
			//cpy Bx to sh_bx
			//row = (realw *)((char *)Bx.ptr + ( k * M + iy ) * Bx.pitch);
			volatile unsigned int indx=( k * M + iy ) * (L + 1) + ix;
			sh_bx[threadIdx.y][threadIdx.x]= Bx[indx];			// Bx[k][j][i], only read once, no need to load to cache	
			if(threadIdx.x == BLOCKSIZE2x-1 || ix == L-1)
				sh_bx[threadIdx.y][threadIdx.x+1]= get_global_cr( &Bx[indx+1] );			// Bx[k][j][i+1], need by another block, load to cache								
			
			//cpy By to sh_by
			//row = (realw *)((char *)By.ptr + ( k * (M+1) + iy ) * By.pitch);
			//indx= ( k * (M+1) + iy ) * L + ix;
			sh_by[threadIdx.y][threadIdx.x]= By[( k * (M+1) + iy ) * L + ix];			// By[k][j][i], only read once, no need to load to cache	
			if(threadIdx.y == BLOCKSIZE2y1-1 || iy == M-1)
			{
				//row = (realw *)((char *)By.ptr + ( k * (M+1) + iy + 1) * By.pitch);
				sh_by[threadIdx.y+1][threadIdx.x]= get_global_cr( &By[( k * (M+1) + iy + 1) * L + ix] );			// By[k][j+1][i], need by another block, load to cache
			}		

			__syncthreads( );
			
			// reuse Bz[k+1][j][i]
			bz0 += Grid_dz(d_mp,k) * ( (sh_bx[threadIdx.y][threadIdx.x+1] - sh_bx[threadIdx.y][threadIdx.x])/dx
										+ (sh_by[threadIdx.y+1][threadIdx.x] - sh_by[threadIdx.y][threadIdx.x])/dy );			
			// Bz[k][j][i]
			//row = (realw *)((char *)Bz.ptr + ( k * M + iy ) * Bz.pitch);	 
			Bz[( k * M + iy )*L + ix] = bz0;
			
		}
#if MAXDEBUG == 1
	if(isnan(bz0))
	{
		printf("error in Bz[0][%d][%d]\n",iy,ix);
		printf("sh_bx:%e - %e\n",sh_bx[threadIdx.y][threadIdx.x+1], sh_bx[threadIdx.y][threadIdx.x]);
		printf("sh_by:%e - %e\n",sh_by[threadIdx.y+1][threadIdx.x], sh_by[threadIdx.y][threadIdx.x]);
	}	
#endif
	}
}



int Bderivate(DArrays *DPtrs, ModelPara MP, ModelPara *d_mp, ModelGrid *d_MG, realw dt, cudaStream_t *stream)
{
  int L,M;
  
  L=MP.L;
	M=MP.M;
	
	//copy to device memory
	//setConst_mp(MP);
	//setConst_dt(dt);
	
	dim3 grid;
	dim3 threads(BLOCKSIZE2x,BLOCKSIZE2y2);
	
	//Bx
	grid.x = (L -1 + BLOCKSIZE2x-1)/BLOCKSIZE2x;
	grid.y = (M + BLOCKSIZE2y2-1)/BLOCKSIZE2y2;
	TRACE("Update Bx");
	UpdateBx<<<grid, threads, 0, stream[0]>>>( DPtrs->Bx, DPtrs->Ey, DPtrs->Ez, d_mp, dt );
	GPU_ERROR_CHECKING("Update Bx");

	cudaEvent_t Event;
	print_CUDA_error_if_any( cudaEventCreateWithFlags( &Event, cudaEventDisableTiming ), 510061 );
	// ANCHOR Event in stream[0]
	print_CUDA_error_if_any( cudaEventRecord(Event, stream[0]), 510062 );
	
	//By
	grid.x = (L + BLOCKSIZE2x-1)/BLOCKSIZE2x;
	grid.y = (M - 1 + BLOCKSIZE2y2-1)/BLOCKSIZE2y2;
	TRACE("Update By");
	UpdateBy<<<grid, threads, 0, stream[1]>>>( DPtrs->By, DPtrs->Ex, DPtrs->Ez, d_mp, dt );
	GPU_ERROR_CHECKING("Update By");
	
	// ANCHOR synchronisation, any new task in stream[0] and steam[1] will wait until the operations on Ez finished
	print_CUDA_error_if_any( cudaStreamWaitEvent(stream[1], Event, 0), 510063 );

	//Bz
	threads.y=BLOCKSIZE2y1;
	grid.x = (L + BLOCKSIZE2x-1)/BLOCKSIZE2x;
	grid.y = (M + BLOCKSIZE2y1-1)/BLOCKSIZE2y1;
	TRACE("Update Bz");
	UpdateBz<<<grid, threads, 0, stream[1]>>>( DPtrs->Bz, DPtrs->Bx, DPtrs->By, d_mp );
	GPU_ERROR_CHECKING("Update Bz");
	
	print_CUDA_error_if_any( cudaEventDestroy( Event ), 510064 );

  return 0;
}


