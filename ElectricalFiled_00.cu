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
/*caculate the Electrical filed from Cur*magnetic field*/
#include"FTDT.h"
#include<math.h>
#include<stdio.h>
#include"Cuda_device.h"
//#include"prepare_constants_cuda.h"

__constant__ realw miu_=4*PI*1.e-7;

// calculate the grid size in x direction
inline __device__ realw Grid_dx( const ModelPara *d_mp, const int i )
{
	realw dx;
	// note i here equals to i in the CPU code
	if( i >= d_mp->ALx2 + d_mp->ALx1 && i < d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 )
		dx = d_mp->dxmin;
	else if( i >= d_mp->ALx2 && i < d_mp->ALx2 + d_mp->ALx1 )
		dx = d_mp->dxmin *MyPow( d_mp->arg1 , d_mp->ALx2 + d_mp->ALx1 -i );
	else if( i < d_mp->ALx2 )
		dx = d_mp->dxmin * MyPow( d_mp->arg1 , d_mp->ALx1 ) * MyPow( d_mp->arg2 , d_mp->ALx2 -i );
	else if( i >= d_mp->Nx + d_mp->ALx1 + d_mp->ALx2 && i < 	d_mp->Nx + d_mp->ARx1 + d_mp->ALx1 + d_mp->ALx2 )
		dx = d_mp->dxmin *MyPow( d_mp->arg1 , i - d_mp->Nx - d_mp->ALx1 - d_mp->ALx2 + 1 );
	else
		dx = d_mp->dxmin * MyPow( d_mp->arg1 , d_mp->ARx1 ) * MyPow( d_mp->arg2 , i - d_mp->Nx - d_mp->ARx1 - d_mp->ALx1 - d_mp->ALx2 + 1 );
		
	return dx;		
}

// calculate the grid size in y direction
inline __device__ realw Grid_dy( const ModelPara *d_mp, const int j )
{
	realw dy;
	// note j here equals to j in the CPU code
	if( j >= d_mp->Ay2 + d_mp->Ay1 && j < d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 )
		dy = d_mp->dymin;
	else if( j >= d_mp->Ay2 && j < d_mp->Ay2 + d_mp->Ay1 )
		dy = d_mp->dymin *MyPow( d_mp->arg1 , d_mp->Ay2 + d_mp->Ay1 -j );
	else if( j < d_mp->Ay2 )
		dy = d_mp->dymin * MyPow( d_mp->arg1 , d_mp->Ay1 ) * MyPow( d_mp->arg2 , d_mp->Ay2 -j );
	else if( j >= d_mp->Ny + d_mp->Ay1 + d_mp->Ay2 && j < 	d_mp->Ny + 2 * d_mp->Ay1 + d_mp->Ay2 )
		dy = d_mp->dymin *MyPow( d_mp->arg1 , j - d_mp->Ny - d_mp->Ay1 - d_mp->Ay2 + 1 );
	else
		dy = d_mp->dymin * MyPow( d_mp->arg1 , d_mp->Ay1 ) * MyPow( d_mp->arg2 , j - d_mp->Ny - 2*d_mp->Ay1 - d_mp->Ay2 + 1 );
		
	return dy;		
}

// calculate the grid size in z direction
inline __device__ realw Grid_dz( const ModelPara *d_mp, const int i )
{
	realw dz;
	// note i here equals to k in the CPU code
	if( i < d_mp->Nz )
		dz = d_mp->dzmin;
	else if( i >= d_mp->Nz && i < d_mp->Nz + d_mp->Az1 )
		dz = d_mp->dzmin *MyPow( d_mp->arg1 , i- d_mp->Nz + 1 );
	else
		dz = d_mp->dzmin * MyPow( d_mp->arg1 , d_mp->Az1 ) * MyPow( d_mp->arg2 , i - d_mp->Nz - d_mp->Az1 + 1 );
		
	return dz;		
}

// the last two parameters are passing by value
__global__ void UpdateEx( cudaPitchedPtr Ex, cudaPitchedPtr By, cudaPitchedPtr Bz, 
													cudaPitchedPtr Con, realw * ByAir, const ModelPara *d_mp,
													const realw dt, const realw lambda) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	//const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-1]; iy in [0,M-2]; iz in [0,N-1]
	// ix=i, iy=j-1, iz=k; transform to index in cpu code
	if( ix < d_mp->L && iy < M-1)
	{
		
		volatile unsigned int k;
		
		//const realw dt=*d_dt;
		//const realw lambda=*d_lambda;
		//----------------------starting with iz=0------------------------------------------
		// determine dy[j] and dy[j-1]
		const realw dy2 = ( Grid_dy(d_mp,iy) + Grid_dy(d_mp,iy+1) );
		// read the conductivity values
		realw *row = (realw *)((char *)Con.ptr + ( iy ) * Con.pitch);
		field sita = Make_field( get_global_cr( &row[ix] ), zerof ); //Cond[0][j-1][i]
		row = (realw *)((char *)Con.ptr + ( iy +1 ) * Con.pitch );
		sita.x= 0.5f*( sita.x + get_global_cr( &row[ix] ) ); //Cond[0][j][i] 
		
		// read in Bz, there is no need to save Bz as temp variable
		row = (realw *)((char *)Bz.ptr + ( iy ) * Bz.pitch);
		realw term1=get_global_cr( &row[ix] );								// Bz[0][j-1][i]
		
		row = (realw *)((char *)Bz.ptr + ( iy + 1 ) * Bz.pitch); 
		term1=( get_global_cr( &row[ix] ) - term1 )/dy2;					  //Bz[0][j][i]
		
		// XXX consider using texture memory in a feature version for ByAir
		row = (realw *)((char *)By.ptr + ( iy + 1 ) * By.pitch);	
		field by2 = Make_field( get_global_cr( &row[ix] ), zerof );//By[0][j][i]		
		
		row = (realw *)((char *)ByAir + ( iy + 1 ) * By.pitch);	 
		by2.y = get_global_cr( &row[ix] );								// ByAir[j][i] 
		
		
		// Ex[0][j][i]
		row = (realw *)((char *)Ex.ptr + ( iy + 1 ) * Ex.pitch);	
		row[ix] = (2.0f*lambda-ave(sita)*dt)/(2.0f*lambda + ave(sita)*dt)*row[ix]+
                        4.0f*dt/(2.0f*lambda + ave(sita)*dt)*( term1 - diff(by2)/(2.0f * Grid_dz(d_mp,0)) )/miu_;
    const int N=d_mp->N;                    
		for( k=1; k < N; k++)
		{
			sita.y = sita.x;
			row = (realw *)((char *)Con.ptr + ( k * M + iy ) * Con.pitch); 
			sita.x = get_global_cr( &row[ix] );            //Cond[k][j-1][i]
			row = (realw *)((char *)Con.ptr + ( k * M + iy + 1 ) * Con.pitch); 
			sita.x= 0.5f*( sita.x + get_global_cr( &row[ix] ) ); //Cond[k][j][i]
			
			// read in Bz, there is no need to save Bz as temp variable
			row = (realw *)((char *)Bz.ptr + ( k * M + iy ) * Bz.pitch);
			term1=get_global_cr( &row[ix] );								// Bz[k][j-1][i]
			
			row = (realw *)((char *)Bz.ptr + ( k * M + iy + 1 ) * Bz.pitch); 
			term1=( get_global_cr( &row[ix] ) - term1 )/dy2;					  //Bz[k][j][i]
			
			//read in By
			by2.y=by2.x;
			row = (realw *)((char *)By.ptr + ( k * (M+1) + iy + 1 ) * By.pitch);	
			by2.x= get_global_cr( &row[ix] );										//By[k][j][i]
			
			// Ex[k][j][i]
			row = (realw *)((char *)Ex.ptr + ( k * (M+1) + iy + 1 ) * Ex.pitch);	 
			row[ix] = (2.0f*lambda-ave(sita)*dt)/(2.0f*lambda + ave(sita)*dt)*row[ix]+
                        4.0f*dt/(2.0f*lambda + ave(sita)*dt)*( term1 - diff(by2)/(Grid_dz(d_mp,k) + Grid_dz(d_mp,k-1)) )/miu_;
			
		}
		
	}
}

// the last two parameters are passing by value
__global__ void UpdateEy( cudaPitchedPtr Ey, cudaPitchedPtr Bx, cudaPitchedPtr Bz, 
													cudaPitchedPtr Con, realw * BxAir, const ModelPara *d_mp,
													const realw dt, const realw lambda ) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	//const int L=d_mp->L;
	//const int M=d_mp->M;
	// ix in [0,L-2]; iy in [0,M-1]; iz in [0,N-1]
	// ix=i-1, iy=j, iz=k; transform to index in cpu code
	if( ix < d_mp->L -1 && iy < d_mp->M )
	{
		//const int N=d_mp->N;
		volatile unsigned int k;
		
		//const realw dt=*d_dt;
		//const realw lambda=*d_lambda;
		// check the size of shared mem
		// this will need shared mem of size 129*8*3=3096 for double procession
		//																	 				 1548 for single procession
		// Probably we could change blocksize to 256 for single procession																
		__shared__ realw sh_con0[BLOCKSIZE1+1];
		__shared__ realw sh_con1[BLOCKSIZE1+1];		// move this to a later stage
		__shared__ realw sh_bz[BLOCKSIZE1+1];				
		
		//----------------------starting with iz=0------------------------------------------
		//cpy con to sh_con0
		realw *row = (realw *)((char *)Con.ptr +  iy * Con.pitch);
		sh_con0[threadIdx.x]=row[ix];
		if (threadIdx.x==BLOCKSIZE1-1 || ix==d_mp->L-2 )
			sh_con0[threadIdx.x+1] = get_global_cr( &row[ix+1] );
			
		//cpy Bz to sh_bz
		row = (realw *)((char *)Bz.ptr +  iy * Bz.pitch);	
		sh_bz[threadIdx.x]=row[ix];
		if (threadIdx.x==BLOCKSIZE1-1 || ix== d_mp->L-2)
			sh_bz[threadIdx.x+1]=get_global_cr( &row[ix+1] );
		
		__syncthreads( );
		
		// calculate dx2
		const realw dx2 = ( Grid_dx(d_mp,ix) + Grid_dx(d_mp,ix+1) );
		
		// XXX consider using texture memory in a feature version for BxAir
		row = (realw *)((char *)Bx.ptr + ( iy ) * Bx.pitch);	
		field bx2 = Make_field( row[ix+1], zerof );//Bx[0][j][i]		
		
		row = (realw *)((char *)BxAir + ( iy ) * Bx.pitch);	 
		bx2.y = row[ix+1];								// BxAir[j][i] 
		
		realw sita = 0.25f * ( sh_con0[threadIdx.x] + sh_con0[threadIdx.x + 1] );
		// Ey[0][j][i]
		row = (realw *)((char *)Ey.ptr + ( iy ) * Ey.pitch);	
		row[ix+1] = (2.0f*lambda - sita*dt)/(2.0f*lambda +  sita * dt)*row[ix+1]+
                        4.0f*dt/(2.0f*lambda +  sita * dt)*( diff(bx2)/(2.0f * Grid_dz(d_mp,0)) -
                        ( sh_bz[threadIdx.x + 1] - sh_bz[threadIdx.x] )/dx2 )/miu_;
                        
		for(k=1;k < (d_mp->N);k++)
		{
			// cpy sh_con0 to sh_con1
			sh_con1[threadIdx.x]=sh_con0[threadIdx.x];
			if (threadIdx.x==BLOCKSIZE1-1 || ix==d_mp->L-2)
				sh_con1[threadIdx.x+1]=sh_con0[threadIdx.x+1];
			__syncthreads( );
			
			// cpy con to sh_con0
			row = (realw *)((char *)Con.ptr + ( k * (d_mp->M) + iy ) * Con.pitch);
			sh_con0[threadIdx.x]=row[ix];
			if (threadIdx.x==BLOCKSIZE1-1 || ix==d_mp->L-2)
				sh_con0[threadIdx.x+1]=get_global_cr( &row[ix+1] );
				
			//cpy Bz to sh_bz
			row = (realw *)((char *)Bz.ptr +  (k* (d_mp->M) +iy) * Bz.pitch);	
			sh_bz[threadIdx.x]=row[ix];
			if (threadIdx.x==BLOCKSIZE1-1 || ix==d_mp->L-2)
				sh_bz[threadIdx.x+1]=get_global_cr( &row[ix+1] );
			
			__syncthreads( );
				
			bx2.y=bx2.x;
			
			row = (realw *)((char *)Bx.ptr + ( k * (d_mp->M) + iy ) * Bx.pitch);	
			bx2.x = row[ix+1];		//Bx[k][j][i]	
			
			sita = 0.25f * ( sh_con0[threadIdx.x] + sh_con0[threadIdx.x + 1] + sh_con1[threadIdx.x] + sh_con1[threadIdx.x +1] );
			
			// Ey[k][j][i]
			row = (realw *)((char *)Ey.ptr + ( k * (d_mp->M) + iy ) * Ey.pitch);	 
      row[ix+1] = (2.0f*lambda - sita*dt)/(2.0f*lambda +  sita * dt)*row[ix+1]+
                        4.0f*dt/(2.0f*lambda +  sita * dt)*( diff(bx2)/( Grid_dz(d_mp,k) + Grid_dz(d_mp,k-1) ) -
                        ( sh_bz[threadIdx.x + 1] - sh_bz[threadIdx.x] )/( dx2 ) )/miu_;                  
			
		}
		
	}
}

// the last two parameters are passing by value
__global__ void UpdateEz( cudaPitchedPtr Ez, cudaPitchedPtr Bx, cudaPitchedPtr By, 
													cudaPitchedPtr Con, const ModelPara *d_mp,
													const realw dt, const realw lambda ) 
{
	volatile unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	volatile unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	
	//const int L=d_mp->L;
	const int M=d_mp->M;
	// ix in [0,L-2]; iy in [0,M-2]; iz in [0,N-1]
	// ix=i-1, iy=j-1, iz=k; transform to index in cpu code
	if( ix < d_mp->L-1 && iy < M-1 )
	{
		//const int N=d_mp->N;
		volatile unsigned int k;
		//const realw dt=*d_dt;
		//const realw lambda=*d_lambda;
		// check the size of shared mem
		// this will need shared mem of size 129*8*3=3096 for double procession
		//																	 				 1548 for single procession
		// Probably we could change blocksize to 256 for single procession																
		__shared__ realw sh_con[2][BLOCKSIZE1+1];
		__shared__ realw sh_by[BLOCKSIZE1+1];				
		realw *row;
		realw sita, term2;
		
		// calculate dx2, dy2
		const realw dx2 = ( Grid_dx(d_mp,ix) + Grid_dx(d_mp,ix+1) );
		const realw dy2 = ( Grid_dy(d_mp,iy) + Grid_dy(d_mp,iy+1) );
		                        
		for(k=0;k < d_mp->N;k++)
		{
			// cpy con to sh_con
			#pragma unroll
			for(int p=0; p < 2; p++)
			{
				row = (realw *)((char *)Con.ptr + ( k * M + iy + p ) * Con.pitch);
				sh_con[p][threadIdx.x]=get_global_cr(&row[ix]);
				if (threadIdx.x== (BLOCKSIZE1-1) || ix== (d_mp->L-2))
					sh_con[p][threadIdx.x+1]=get_global_cr(&row[ix+1]);
			}
			
			//cpy By to sh_by
			row = (realw *)((char *)By.ptr +  (k*(M+1)+iy+1) * By.pitch);	
			sh_by[threadIdx.x]=row[ix];
			if (threadIdx.x==BLOCKSIZE1-1 || ix==d_mp->L-2)
				sh_by[threadIdx.x+1]=get_global_cr(&row[ix+1]);
			
			__syncthreads( );
				
			// read bx
			row = (realw *)((char *)Bx.ptr + ( k * M + iy ) * Bx.pitch);	
			term2 = -get_global_cr(&row[ix+1]);		//-Bx[k][j-1][i]	
			
			row = (realw *)((char *)Bx.ptr + ( k * M + iy +1 ) * Bx.pitch);	
			term2 += get_global_cr(&row[ix+1]);		//Bx[k][j][i]-Bx[k][j-1][i]
			
			sita = 0.25f * ( sh_con[0][threadIdx.x] + sh_con[0][threadIdx.x + 1] + sh_con[1][threadIdx.x] + sh_con[1][threadIdx.x +1] );
			
			// Ez[k][j][i]
			row = (realw *)((char *)Ez.ptr + ( k * (M+1) + iy + 1 ) * Ez.pitch);	 
      row[ix+1] = (2.0f*lambda - sita*dt)/(2.0f*lambda +  sita * dt)*row[ix+1]+
                        4.0f*dt/(2.0f*lambda +  sita * dt)*(( sh_by[threadIdx.x + 1] - sh_by[threadIdx.x])/( dx2 )
                        -term2/( dy2 ) )/miu_;                  
			
		}
	}
}
// current is passed by value
__global__ void AddTx_x( cudaPitchedPtr Ex, cudaPitchedPtr Con, const ModelPara *d_mp, 
													const TxPos *d_Tx, const realw current )
{
		const int ix = threadIdx.x;
		const int M=d_mp->M;
		const int ixa=d_Tx->ixa;
		const int iya=d_Tx->iya;
		const int iza=d_Tx->iza;
		realw *row;
		realw sita;
		if(ix <= (d_Tx->ixb - ixa)) 
		{
			//read con
			if(iza == 0)
			{
				row = (realw *)((char *)Con.ptr + ( iza * M + iya ) * Con.pitch);
				sita = row[ix+ixa];				//Con[k][j][i]
				row = (realw *)((char *)Con.ptr + ( iza * M + iya - 1) * Con.pitch);
				sita += row[ix+ixa];				//Con[k][j-1][i]
			
			}
			else
			{
				row = (realw *)((char *)Con.ptr + ( iza * M + iya ) * Con.pitch);
				sita = row[ix+ixa];				//Con[k][j][i]
				row = (realw *)((char *)Con.ptr + ( iza * M + iya - 1) * Con.pitch);
				sita += row[ix+ixa];				//Con[k][j-1][i]
				row = (realw *)((char *)Con.ptr + ( (iza -1) * M + iya ) * Con.pitch);
				sita += row[ix+ixa];				//Con[k-1][j][i]
				row = (realw *)((char *)Con.ptr + ( (iza -1) * M + iya -1 ) * Con.pitch);
				sita += row[ix+ixa];				//Con[k-1][j-1][i]
			}
			
			row = (realw *)((char *)Ex.ptr + ( iza * (M+1) + iya ) * Ex.pitch);
			row[ix+ixa]-=2.0f*current/( 0.25f * sita * Grid_dy(d_mp,iya) * Grid_dz(d_mp,iza));
		}
}

// last parameter is passed by value
__global__ void AddTx_y( cudaPitchedPtr Ey, cudaPitchedPtr Con, const ModelPara *d_mp, 
													const TxPos *d_Tx, const realw current )
{
		const int iy = threadIdx.x;
		const int M=d_mp->M;
		realw *row;
		realw sita;
		
		const int ixa=d_Tx->ixa;
		const int iya=d_Tx->iya;
		const int iza=d_Tx->iza;
		
		if(iy <= d_Tx->iyb - iya)
		{
			//read con
			if(iza == 0)
			{
				row = (realw *)((char *)Con.ptr + ( iza * M + iy + iya ) * Con.pitch);
				sita = row[ixa-1];				//Con[k][j][i-1]
				sita += row[ixa];				//Con[k][j][i]
			
			}
			else
			{
				row = (realw *)((char *)Con.ptr + ( iza * M + iy + iya ) * Con.pitch);
				sita = row[ixa-1];						//Con[k][j][i-1]
				sita += row[ixa];				//Con[k][j][i]
				row = (realw *)((char *)Con.ptr + ( (iza -1) * M + iy + iya ) * Con.pitch);
				sita += row[ixa-1];				//Con[k-1][j][i-1]
				sita += row[ixa];				//Con[k-1][j][i]
			}
			
			row = (realw *)((char *)Ey.ptr + ( iza * M + iy + iya ) * Ey.pitch);
			row[ixa]-=2.0f*current/( 0.25f * sita * Grid_dx(d_mp,ixa) * Grid_dz(d_mp,iza));
		}
}

__global__ void AddTx_z( cudaPitchedPtr Ez, cudaPitchedPtr Con, const ModelPara *d_mp, 
													const TxPos *d_Tx, const realw current )
{
		const int iz = threadIdx.x;
		const int M=d_mp->M;
		const int ixa=d_Tx->ixa;
		const int iya=d_Tx->iya;
		const int iza=d_Tx->iza;
		
		realw *row;
		realw sita;
		
		if(iz <= d_Tx->izb - iza)
		{
			//read con
			row = (realw *)((char *)Con.ptr + ( ( iza + iz )* M + iya ) * Con.pitch);
			sita = row[ixa-1];						//Con[k][j][i-1]
			sita += row[ixa];						//Con[k][j][i]
			row = (realw *)((char *)Con.ptr + ( (iza + iz) * M + iya -1) * Con.pitch);
			sita += row[ixa-1];				//Con[k][j-1][i-1]
			sita += row[ixa];				//Con[k][j-1][i]
			
			row = (realw *)((char *)Ez.ptr + ( (iza + iz)* (M+1) + iya ) * Ez.pitch);
			row[ixa]-=2.0f*current/( 0.25f * sita * Grid_dx(d_mp,ixa) * Grid_dy(d_mp,iya));
		}
}


int ElectricalField( DArrays *DPtrs, ModelPara *MP, ModelPara *d_mp,
                   	ModelGrid *MG,	realw lambda,	realw dt,	realw wave,	
                   	TxPos xyzTx, TxPos *d_Tx)
{
  int L,M,N;
  						
	L=MP->L;
	M=MP->M;
	N=MP->N;
	
	dim3 grid;
	dim3 threads(BLOCKSIZE1,1);
	
	//Ex
	grid.x = (L + BLOCKSIZE1-1)/BLOCKSIZE1;
	grid.y = M-1;
	TRACE("Update Ex");
	UpdateEx<<<grid,threads>>>( DPtrs->Ex, DPtrs->By, DPtrs->Bz, DPtrs->Con, DPtrs->ByAir, d_mp, dt, lambda );
	GPU_ERROR_CHECKING("Update Ex");
	
	//Ey
	grid.x = (L-1 + BLOCKSIZE1-1)/BLOCKSIZE1;
	grid.y = M;
	TRACE("Update Ey");
	UpdateEy<<<grid,threads>>>( DPtrs->Ey, DPtrs->Bx, DPtrs->Bz, DPtrs->Con, DPtrs->BxAir, d_mp, dt, lambda );
	GPU_ERROR_CHECKING("Update Ey");
	
	//Ez
	grid.x = (L-1 + BLOCKSIZE1-1)/BLOCKSIZE1;
	grid.y = M-1;
	TRACE("Update Ez");
	UpdateEz<<<grid,threads>>>( DPtrs->Ez, DPtrs->Bx, DPtrs->By, DPtrs->Con, d_mp, dt, lambda );
	GPU_ERROR_CHECKING("Update Ez");
  
  // add transmitter source
  if(wave!=0)
  {
		// first need to cpy Tx information to constant mem
		switch( xyzTx.ort )
		{
		case 1 :
			wave/=(MG->X_Bzold[xyzTx.ixb]-MG->X_Bzold[xyzTx.ixa]+MG->dx[xyzTx.ixb]/2+MG->dx[xyzTx.ixa]/2);
			//setConst_Tx(xyzTx.ixa, xyzTx.ixb, xyzTx.iya, xyzTx.iyb, xyzTx.iza, xyzTx.izb, wave);
			threads.x=xyzTx.ixb-xyzTx.ixa+1;
			TRACE("Add Tx x");
			AddTx_x<<<1,threads>>>( DPtrs->Ex, DPtrs->Con, d_mp, d_Tx, wave );
			GPU_ERROR_CHECKING("Add Tx x");			
			break;
		case 2 :
			wave/=(MG->Y_Bzold[xyzTx.iyb]-MG->Y_Bzold[xyzTx.iya]+MG->dy[xyzTx.iyb]/2+MG->dy[xyzTx.iya]/2);
			//setConst_Tx(xyzTx.ixa, xyzTx.ixb, xyzTx.iya, xyzTx.iyb, xyzTx.iza, xyzTx.izb, wave);
			threads.x=xyzTx.iyb-xyzTx.iya+1;
			TRACE("Add Tx y");
			AddTx_y<<<1,threads>>>( DPtrs->Ey, DPtrs->Con, d_mp, d_Tx, wave );
			GPU_ERROR_CHECKING("Add Tx y");		
			break;
		case 3 :
			wave/=(MG->Z[xyzTx.izb]-MG->Z[xyzTx.iza]+MG->dz[xyzTx.izb]/2+MG->dz[xyzTx.iza]/2);
			//setConst_Tx(xyzTx.ixa, xyzTx.ixb, xyzTx.iya, xyzTx.iyb, xyzTx.iza, xyzTx.izb, wave);
			threads.x=xyzTx.izb-xyzTx.iza+1;
			TRACE("Add Tx z");
			AddTx_z<<<1,threads>>>( DPtrs->Ez, DPtrs->Con, d_mp, d_Tx, wave );
			GPU_ERROR_CHECKING("Add Tx z");		
			break;
		default :
			printf("Current only support x,y,or z oriented bipole source\n");
		}	
 	}
  return 0;

}
