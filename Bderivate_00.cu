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
__global__ void UpdateBx( cudaPitchedPtr Bx, cudaPitchedPtr Ey, cudaPitchedPtr Ez,
													const ModelPara *d_mp, const realw dt ) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	//const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-2]; iy in [0,M-1]; iz in [0,N-1]
	// ix=i-1, iy=j, iz=k; transform to index in cpu code
	if( ix < d_mp->L-1 && iy < M )
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
		realw *row = (realw *)((char *)Ey.ptr + ( iy ) * Ey.pitch);	
		field ey2 = Make_field( row[ix+1], zerof );//Ey[0][j][i]		
		
		for(k=0;k<d_mp->N;k++)
		{
			//cpy Ez to sh_ez
			row = (realw *)((char *)Ez.ptr + ( k * (M+1) + iy ) * Ez.pitch);
			sh_ez[threadIdx.y][threadIdx.x]= row[ix+1];			// Ez[k][j][i], only read once, no need to load to cache	
			if(threadIdx.y == BLOCKSIZE2y2-1 || iy == M-1)
			{
				row = (realw *)((char *)Ez.ptr + ( k * (M+1) + iy+1 ) * Ez.pitch);	
				sh_ez[threadIdx.y+1][threadIdx.x]= get_global_cr( &row[ix+1] );			// Ez[k][j+1][i], need by another block, load to cache								
			}
			if(ix==0 && iy==0)
				dz=Grid_dz(d_mp,k);
				
			__syncthreads( );
			
			//read in Ey
			ey2.y=ey2.x;
			row = (realw *)((char *)Ey.ptr + ( (k+1) * M + iy ) * Ey.pitch);	
			ey2.x= row[ix+1];										//Ey[k+1][j][i], read only once
			
			// Bx[k][j][i]
			row = (realw *)((char *)Bx.ptr + ( k * M + iy ) * Bx.pitch);	 
			//row[ix+1] += ( diff(ey2)/Grid_dz(d_mp,k) - 
			//								(sh_ez[threadIdx.y+1][threadIdx.x] - sh_ez[threadIdx.y][threadIdx.x])/dy )*dt;
			row[ix+1] += ( diff(ey2)/dz - 
											(sh_ez[threadIdx.y+1][threadIdx.x] - sh_ez[threadIdx.y][threadIdx.x])/dy )*dt;
			
		}
		
	}
}

__global__ void UpdateBy( cudaPitchedPtr By, cudaPitchedPtr Ex, cudaPitchedPtr Ez,
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
		realw *row = (realw *)((char *)Ex.ptr + (iy + 1) * Ex.pitch);	
		field ex2 = Make_field( row[ix], zerof );//Ex[0][j][i]		
		
		for(k=0;k<d_mp->N;k++)
		{
			//cpy Ez to sh_ez
			row = (realw *)((char *)Ez.ptr + ( k * (M+1) + iy + 1) * Ez.pitch);
			sh_ez[threadIdx.y][threadIdx.x]= row[ix];			// Ez[k][j][i], only read once, no need to load to cache	
			if(threadIdx.x == BLOCKSIZE2x-1 || ix == L-1)
				sh_ez[threadIdx.y][threadIdx.x+1]= get_global_cr( &row[ix+1] );			// Ez[k][j][i+1], need by another block, load to cache								
			
			if(ix==0 && iy==0)
				dz=Grid_dz(d_mp,k);
				
			__syncthreads( );
			
			//read in Ex
			ex2.y=ex2.x;
			row = (realw *)((char *)Ex.ptr + ( (k+1) * (M+1) + iy + 1) * Ex.pitch);	
			ex2.x= row[ix];										//Ey[k+1][j][i], read only once
			
			// By[k][j][i]
			row = (realw *)((char *)By.ptr + ( k * (M+1) + iy + 1) * By.pitch);	 
			row[ix] += ( (sh_ez[threadIdx.y][threadIdx.x+1] - sh_ez[threadIdx.y][threadIdx.x])/dx - 
											 diff(ex2)/dz )*dt;
			
		}
		
	}
}

__global__ void UpdateBz( cudaPitchedPtr Bz, cudaPitchedPtr Bx, cudaPitchedPtr By, const ModelPara *d_mp) 
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
		volatile unsigned int k;
		realw bz0=0.0f;	//reuse bz
		realw *row;
		// smem of size 17*16*2*8=4353B
		__shared__ realw sh_bx[BLOCKSIZE2y1][BLOCKSIZE2x+1];
		__shared__ realw sh_by[BLOCKSIZE2y1+1][BLOCKSIZE2x];
		//__shared__ realw dx;
		//__shared__ realw dy;
		// dx[i] and dy[j] keep constant for different z
		const realw dx = Grid_dx(d_mp,ix);
		const realw dy = Grid_dy(d_mp,iy);
		
		// Bz is calculated from bottom up		
		for(k=d_mp->N-1;k>=0;k--)
		{
			//cpy Bx to sh_bx
			row = (realw *)((char *)Bx.ptr + ( k * M + iy ) * Bx.pitch);
			sh_bx[threadIdx.y][threadIdx.x]= row[ix];			// Bx[k][j][i], only read once, no need to load to cache	
			if(threadIdx.x == BLOCKSIZE2x-1 || ix == L-1)
				sh_bx[threadIdx.y][threadIdx.x+1]= get_global_cr( &row[ix+1] );			// Bx[k][j][i+1], need by another block, load to cache								
			
			//cpy By to sh_by
			row = (realw *)((char *)By.ptr + ( k * (M+1) + iy ) * By.pitch);
			sh_by[threadIdx.y][threadIdx.x]= row[ix];			// By[k][j][i], only read once, no need to load to cache	
			if(threadIdx.y == BLOCKSIZE2y1-1 || iy == M-1)
			{
				row = (realw *)((char *)By.ptr + ( k * (M+1) + iy + 1) * By.pitch);
				sh_by[threadIdx.y+1][threadIdx.x]= get_global_cr( &row[ix] );			// By[k][j+1][i], need by another block, load to cache
			}		

			__syncthreads( );
			
			// reuse Bz[k+1][j][i]
			bz0 += Grid_dz(d_mp,k) * ( (sh_bx[threadIdx.y][threadIdx.x+1] - sh_bx[threadIdx.y][threadIdx.x])/dx
										+ (sh_by[threadIdx.y+1][threadIdx.x] - sh_by[threadIdx.y][threadIdx.x])/dy );
			
			// Bz[k][j][i]
			row = (realw *)((char *)Bz.ptr + ( k * M + iy ) * Bz.pitch);	 
			row[ix] = bz0;
			
		}
	}
}



int Bderivate(DArrays *DPtrs, ModelPara MP, ModelPara *d_mp, ModelGrid *d_MG, realw dt)
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
	UpdateBx<<<grid,threads>>>( DPtrs->Bx, DPtrs->Ey, DPtrs->Ez, d_mp, dt );
	GPU_ERROR_CHECKING("Update Bx");
	
	//By
	grid.x = (L + BLOCKSIZE2x-1)/BLOCKSIZE2x;
	grid.y = (M - 1 + BLOCKSIZE2y2-1)/BLOCKSIZE2y2;
	TRACE("Update By");
	UpdateBy<<<grid,threads>>>( DPtrs->By, DPtrs->Ex, DPtrs->Ez, d_mp, dt );
	GPU_ERROR_CHECKING("Update By");
	
	//Bz
	threads.y=BLOCKSIZE2y1;
	grid.x = (L + BLOCKSIZE2x-1)/BLOCKSIZE2x;
	grid.y = (M + BLOCKSIZE2y1-1)/BLOCKSIZE2y1;
	TRACE("Update Bz");
	UpdateBz<<<grid,threads>>>( DPtrs->Bz, DPtrs->Bx, DPtrs->By, d_mp );
	GPU_ERROR_CHECKING("Update Bz");
	
  return 0;
}


