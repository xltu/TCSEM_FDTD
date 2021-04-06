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
 /*------------------ Read in the Tx, Rx parameters from the input files-----------------------------
 @XT @March.22.201, support Bipole transmitter of arbitrary orentation
										support a transmitter of two bipoles in arbitrary orentation 
  */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include"FTDT.h"

int LoadRxTxPos(ModelPara MP, ModelGrid MG, RxTime *tRx, RxPos *xyzRx, TxPos **xyzTx, int *NRx, int *NTx)
{
	TRACE("Loading Rx and Tx parameters\n");
	FILE *FRxIn,*FTxIn;
	int ir, nr, M, L, K;
	int ij, ort, itx;
	realw x, y, z, azi, dip;
	//------------------------------- load in the receiver file-------------------------------------------------------------------------------
	if((FRxIn=fopen("Receiver.in","r"))==NULL)
	{
		printf("Fail to open the receiver file!\n");
		perror("Error: ");
		getchar();
		return(1);
	}
	
	fscanf(FRxIn,"%lf %lf\n", &(tRx->T0), &(tRx->Tn));	// recording start and end time
	fscanf(FRxIn,"%d %d\n", &(tRx->Nt), &(tRx->IsLog));	// total number of time channels, is the recording time logrithmitically equally spaced
	
	// whether record the Ex, Ey, Ez, Bx, By, Bz component of the field
	fscanf(FRxIn,"%d %d %d %d %d %d\n", &(tRx->isComp[0]), &(tRx->isComp[1]), &(tRx->isComp[2]), &(tRx->isComp[3]), &(tRx->isComp[4]), &(tRx->isComp[5]));
	
	// total number of receivers
	fscanf(FRxIn,"%d\n", NRx);	// total number of receivers
	xyzRx->x=(realw *)malloc((*NRx)*sizeof(realw));
	xyzRx->y=(realw *)malloc((*NRx)*sizeof(realw));
	xyzRx->z=(realw *)malloc((*NRx)*sizeof(realw));
	xyzRx->ix=(int *)malloc((*NRx)*sizeof(int));
	xyzRx->iy=(int *)malloc((*NRx)*sizeof(int));
	xyzRx->iz=(int *)malloc((*NRx)*sizeof(int));
	if( xyzRx->x==NULL || xyzRx->y==NULL || xyzRx->z==NULL || xyzRx->ix==NULL || xyzRx->iy==NULL || xyzRx->iz==NULL )
	{
		printf("Faile to allocate memory for receiver array\n");
		perror("Error: ");
		fclose(FRxIn);
		return(-1);
	}
	for(ir=0;ir<*NRx;ir++)
	{
		if(feof(FRxIn)) 
		{
			printf("Error: only able to read %d receiver posistions in the receiver file, while the total number of receiver is %d\n", ir+1, *NRx);
			fclose(FRxIn);
			exit(-1);
		}
		fscanf(FRxIn,"%d %lf %lf %lf\n", &nr, &x, &y, &z);	// recording start and end time
		xyzRx->x[nr-1]=x;
		xyzRx->y[nr-1]=y;
		xyzRx->z[nr-1]=z;
	}
	
	fclose(FRxIn);
	TRACE("Rx parameters were loaded successfully\n");
	//-------------------------------------------Receiver file successfully loaded--------------------------------------------------------------
	L=MP.L;
	M=MP.M;
	K=MP.N;
	//-------------------------------------------check receiver positions-----------------------------------------------------------------------
	#pragma omp parallel for private(ir,nr)
	for(ir=0;ir<*NRx;ir++)
	{
	 	// x position
	 	nr= searchInsert(MG.X_Bzold, L, xyzRx->x[ir]);
	 	if (nr==0 || nr==L)
	 	{
	 		printf("Error: Receiver %d located outside of modeling domain in x direction \n",ir+1);
	 		exit(-1); 
	 	}
	 	if (nr<10 || nr>L-10)
	 		printf("Warning: Receiver %d located too close to modeling boundary in x direction\n",ir+1);
	 	xyzRx->ix[ir]=nr;
	 	
	 	// y position
	 	nr= searchInsert(MG.Y_Bzold, M, xyzRx->y[ir]);
	 	if (nr==0 || nr==M)
	 	{
	 		printf("Error: Receiver %d located outside of modeling domain in y direction \n",ir+1);
	 		exit(-1); 
	 	}
	 	if (nr<10 || nr>M-10)
	 		printf("Warning: Receiver %d located too close to modeling boundary in y direction\n",ir+1);
	 	xyzRx->iy[ir]=nr;	
	 	
	 	// z position
	 	nr= searchInsert(MG.Z, K, xyzRx->z[ir]);	
	 	xyzRx->iz[ir]=nr;		
	}
	//-------------------------------------------end of checking receiver positions-------------------------------------------------------------
	
	//--------------------------------------------load in transmitter positions-----------------------------------------------------------------
	// !ANCHOR: support arbitrary orientation of the transimitters 
	if((FTxIn=fopen("Transmitter.in","r"))==NULL)
	{
		printf("Fail to open the transmitter file!\n");
		perror("Error: ");
		getchar();
		return(1);
	}
	
	fscanf(FTxIn,"%d\n", NTx);	// number of transmitter positions
	*xyzTx=(TxPos *)malloc((*NTx)*sizeof(TxPos));
	if (*xyzTx==NULL)
	{
		printf("Faile to allocate memory for transmitter array\n");
		perror("Error: ");
		fclose(FTxIn);
		return(-1);
	}
	for(ij=0;ij<*NTx;ij++)
	{
		if(feof(FTxIn)) 
		{
			printf("Error: only able to read %d transmitter posistions in the transmitter file, while the total number of transmitter is %d\n", ij+1, *NTx);
			fclose(FTxIn);
			exit(-1);
		}
		fscanf(FTxIn,"%d %d", &itx, &ort);
		(*xyzTx)[itx-1].ort=ort;

		if( ort<4 ) 
		{ // bipole transmitter oriented in x, y, or z direction
			fscanf( FTxIn," %lf %lf %lf %lf %d", &((*xyzTx)[itx-1].BipoLen), &((*xyzTx)[itx-1].x), &((*xyzTx)[itx-1].y), 
							&((*xyzTx)[itx-1].z), &((*xyzTx)[itx-1].nr) );
		}
		else if( ort==4 )
		{ // bipole transmitter in arbitrary orentation
			fscanf( FTxIn," %lf %lf %lf %lf %lf %lf %d", &((*xyzTx)[itx-1].BipoLen), &((*xyzTx)[itx-1].x), &((*xyzTx)[itx-1].y), 
							&((*xyzTx)[itx-1].z), &azi, &dip, &((*xyzTx)[itx-1].nr) );	
			(*xyzTx)[itx-1].azi = azi*PI/180.0;
			(*xyzTx)[itx-1].dip = dip*PI/180.0;
		}
		else if( ort==5 )
		{
			// two bipole transmitter in arbitrary orentation
			fscanf( FTxIn," %lf %lf %lf %lf %lf %lf", &((*xyzTx)[itx-1].BipoLen), &((*xyzTx)[itx-1].x), &((*xyzTx)[itx-1].y), 
							&((*xyzTx)[itx-1].z), &azi, &dip );
			(*xyzTx)[itx-1].azi = azi*PI/180.0;
			(*xyzTx)[itx-1].dip = dip*PI/180.0;

			fscanf( FTxIn," %lf %lf %lf %lf %lf %lf %d", &((*xyzTx)[itx-1].BipoLen1), &((*xyzTx)[itx-1].x1), &((*xyzTx)[itx-1].y1), 
							&((*xyzTx)[itx-1].z1), &azi, &dip, &((*xyzTx)[itx-1].nr) );	
			(*xyzTx)[itx-1].azi1 = azi*PI/180.0;
			(*xyzTx)[itx-1].dip1 = dip*PI/180.0;							
		}
		else
		{
			printf("Illegal type of transmitter\n");
			perror("Error: error in transmitter file");
			fclose(FTxIn);
			return(-1);
		}

		nr=(*xyzTx)[itx-1].nr;
		(*xyzTx)[itx-1].RxNum=(int *)malloc(nr*sizeof(int));
		if ( (*xyzTx)[itx-1].RxNum == NULL )
		{
			printf("Faile to allocate memory for transmitter array\n");
			perror("Error: ");
			fclose(FTxIn);
			return(-1);
		}
		
		for(ir=0;ir<nr-1;ir++)
			fscanf(FTxIn," %d", (*xyzTx)[itx-1].RxNum+ir);	// corresponding receiver # recorded the current transmitter
		fscanf(FTxIn," %d\n", (*xyzTx)[itx-1].RxNum+ir);	// corresponding receiver # recorded the current transmitter	
	}
	
	fclose(FTxIn);
	//--------------------------------------------Successfully loaded Tx positions--------------------------------------------------------------
	TRACE("Tx parameters were loaded successfully\n");
	//---------------------------------------------------------check transmitter position-------------------------------------------------------
	#pragma omp parallel for if (*NTx > 3) private(ij,nr)
	for(ij=0;ij<*NTx;ij++)
	{
		
		// x direction dipole
		if( (*xyzTx)[ij].ort == 1)
		{
			(*xyzTx)[ij].ixa = searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x-(*xyzTx)[ij].BipoLen/2);
			(*xyzTx)[ij].ixb = searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x+(*xyzTx)[ij].BipoLen/2)-1;
			if ( (*xyzTx)[ij].ixa ==0 || (*xyzTx)[ij].ixb ==L )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in x direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].ixa < 10 || (*xyzTx)[ij].ixb > L-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in x direction\n",ij+1);
		 		
			nr=searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y);
			(*xyzTx)[ij].icty=nr;
			if (nr==0 || nr==M)
		 	{
		 		printf("Error: Transmitter #%d located outside of modeling domain in y direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if (nr<10 || nr>M-10)
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in y direction\n",ij+1);
			
			nr=searchInsert(MG.Z, K, (*xyzTx)[ij].z);
			(*xyzTx)[ij].ictz=nr;
		}	
	
		// y direction dipole	
		else if( (*xyzTx)[ij].ort == 2)
		{
			nr = searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x);
			(*xyzTx)[ij].ictx=nr;
			if ( nr ==0 || nr ==L )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in x direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( nr < 10 || nr > L-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in x direction\n",ij+1);
		 		
		 	(*xyzTx)[ij].iya = searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y-(*xyzTx)[ij].BipoLen/2);
			(*xyzTx)[ij].iyb = searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y+(*xyzTx)[ij].BipoLen/2)-1;
			if ( (*xyzTx)[ij].iya ==0 || (*xyzTx)[ij].iyb ==M )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in y direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].iya < 10 || (*xyzTx)[ij].iyb > M-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in y direction\n",ij+1);	
		 		
		 	nr=searchInsert(MG.Z, K, (*xyzTx)[ij].z);
			(*xyzTx)[ij].ictz=nr;
		 }	
		// z direction dipole 		
		else if( (*xyzTx)[ij].ort == 3 )
		{
			nr = searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x);
			(*xyzTx)[ij].ictx=nr;
			if ( nr ==0 || nr ==L )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in x direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( nr < 10 || nr > L-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in x direction\n",ij+1);
		 		
		 	nr=searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y);
			(*xyzTx)[ij].icty=nr;
			if (nr==0 || nr==M)
		 	{
		 		printf("Error: Transmitter #%d located outside of modeling domain in y direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if (nr<10 || nr>M-10)
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in y direction\n",ij+1);
		 		
		 	(*xyzTx)[ij].iza = searchInsert(MG.Z, K, (*xyzTx)[ij].z-(*xyzTx)[ij].BipoLen/2);
			(*xyzTx)[ij].izb = searchInsert(MG.Z, K, (*xyzTx)[ij].z+(*xyzTx)[ij].BipoLen/2)-1;		
			
		}
		// arbitrary orented bipole
		else if( (*xyzTx)[ij].ort == 4 )
		{
			(*xyzTx)[ij].ixa = searchInsert( MG.X_Bzold, L, 
														(*xyzTx)[ij].x - (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*cos((*xyzTx)[ij].azi)/2.0 );
			(*xyzTx)[ij].ixb = searchInsert(MG.X_Bzold, L, 
														(*xyzTx)[ij].x + (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*cos((*xyzTx)[ij].azi)/2.0 );
			if ((*xyzTx)[ij].ixb > (*xyzTx)[ij].ixa )
					(*xyzTx)[ij].ixb--;
			else if ((*xyzTx)[ij].ixb < (*xyzTx)[ij].ixa )
					(*xyzTx)[ij].ixa--;
			else
			;														
			if ( (*xyzTx)[ij].ixa ==0 || (*xyzTx)[ij].ixb ==L )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in x direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].ixa < 10 || (*xyzTx)[ij].ixb > L-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in x direction\n",ij+1);

			(*xyzTx)[ij].iya = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y - (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*sin((*xyzTx)[ij].azi)/2.0 );
			(*xyzTx)[ij].iyb = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y + (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*sin((*xyzTx)[ij].azi)/2.0 );
			if ((*xyzTx)[ij].iyb > (*xyzTx)[ij].iya )
					(*xyzTx)[ij].iyb--;
			else if ((*xyzTx)[ij].iyb < (*xyzTx)[ij].iya )
					(*xyzTx)[ij].iya--;
			else
			;															
			if ( (*xyzTx)[ij].iya ==0 || (*xyzTx)[ij].iyb ==M )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in y direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].iya < 10 || (*xyzTx)[ij].iyb > M-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in y direction\n",ij+1);

			(*xyzTx)[ij].iza = searchInsert( MG.Z, K, (*xyzTx)[ij].z - (*xyzTx)[ij].BipoLen*sin((*xyzTx)[ij].dip)/2.0 );
			(*xyzTx)[ij].izb = searchInsert( MG.Z, K, (*xyzTx)[ij].z + (*xyzTx)[ij].BipoLen*sin((*xyzTx)[ij].dip)/2.0 );		
			if ((*xyzTx)[ij].izb > (*xyzTx)[ij].iza )
					(*xyzTx)[ij].izb--;
			else if ((*xyzTx)[ij].izb < (*xyzTx)[ij].iza )
					(*xyzTx)[ij].iza--;
			else
			;
			(*xyzTx)[ij].ictx=searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x); 
			(*xyzTx)[ij].icty=searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y);
			(*xyzTx)[ij].ictz=searchInsert(MG.Z, K, (*xyzTx)[ij].z);
		}
		// two arbitrary orented bipole
		else if( (*xyzTx)[ij].ort == 5 )
		{
			(*xyzTx)[ij].ixa = searchInsert( MG.X_Bzold, L, 
														(*xyzTx)[ij].x - (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*cos((*xyzTx)[ij].azi)/2.0 );
			(*xyzTx)[ij].ixb = searchInsert(MG.X_Bzold, L, 
														(*xyzTx)[ij].x + (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*cos((*xyzTx)[ij].azi)/2.0 );
			if ((*xyzTx)[ij].ixb > (*xyzTx)[ij].ixa )
					(*xyzTx)[ij].ixb--;
			else if ((*xyzTx)[ij].ixb < (*xyzTx)[ij].ixa )
					(*xyzTx)[ij].ixa--;
			else
			;
			(*xyzTx)[ij].ixa1 = searchInsert( MG.X_Bzold, L, 
														(*xyzTx)[ij].x1 - (*xyzTx)[ij].BipoLen1*cos((*xyzTx)[ij].dip1)*cos((*xyzTx)[ij].azi1)/2.0 );
			(*xyzTx)[ij].ixb1 = searchInsert( MG.X_Bzold, L, 
														(*xyzTx)[ij].x1 + (*xyzTx)[ij].BipoLen1*cos((*xyzTx)[ij].dip1)*cos((*xyzTx)[ij].azi1)/2.0 );
			if ((*xyzTx)[ij].ixb1 > (*xyzTx)[ij].ixa1 )
					(*xyzTx)[ij].ixb1--;
			else if ((*xyzTx)[ij].ixb1 < (*xyzTx)[ij].ixa1 )
					(*xyzTx)[ij].ixa1--;
			else
			;				
			if ( (*xyzTx)[ij].ixa ==0 || (*xyzTx)[ij].ixa1 == 0 || (*xyzTx)[ij].ixb ==L || (*xyzTx)[ij].ixb1 ==L )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in x direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].ixa < 10 || (*xyzTx)[ij].ixa1 < 10 || (*xyzTx)[ij].ixb > L-10 || (*xyzTx)[ij].ixb1 > L-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in x direction\n",ij+1);

			(*xyzTx)[ij].iya = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y - (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*sin((*xyzTx)[ij].azi)/2.0 );
			(*xyzTx)[ij].iyb = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y + (*xyzTx)[ij].BipoLen*cos((*xyzTx)[ij].dip)*sin((*xyzTx)[ij].azi)/2.0 );
			if ((*xyzTx)[ij].iyb > (*xyzTx)[ij].iya )
					(*xyzTx)[ij].iyb--;
			else if ((*xyzTx)[ij].iyb < (*xyzTx)[ij].iya )
					(*xyzTx)[ij].iya--;
			else
			;						
			(*xyzTx)[ij].iya1 = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y1 - (*xyzTx)[ij].BipoLen1*cos((*xyzTx)[ij].dip1)*sin((*xyzTx)[ij].azi1)/2.0 );
			(*xyzTx)[ij].iyb1 = searchInsert(MG.Y_Bzold, M, 
														(*xyzTx)[ij].y1 + (*xyzTx)[ij].BipoLen1*cos((*xyzTx)[ij].dip1)*sin((*xyzTx)[ij].azi1)/2.0 );
			if ((*xyzTx)[ij].iyb1 > (*xyzTx)[ij].iya1 )
					(*xyzTx)[ij].iyb1--;
			else if ((*xyzTx)[ij].iyb1 < (*xyzTx)[ij].iya1 )
					(*xyzTx)[ij].iya1--;
			else
			;					
			if ( (*xyzTx)[ij].iya ==0 || (*xyzTx)[ij].iya1 ==0 || (*xyzTx)[ij].iyb ==M || (*xyzTx)[ij].iyb1 ==M )
			{
		 		printf("Error: Transmitter #%d located outside of modeling domain in y direction \n",ij+1);
		 		exit(-1); 
		 	}
		 	if ( (*xyzTx)[ij].iya < 10 || (*xyzTx)[ij].iya1 < 10 || (*xyzTx)[ij].iyb > M-10  || (*xyzTx)[ij].iyb1 > M-10 )
		 		printf("Warning: Transmitter #%d located too close to modeling boundary in y direction\n",ij+1);

			(*xyzTx)[ij].iza = searchInsert( MG.Z, K, (*xyzTx)[ij].z - (*xyzTx)[ij].BipoLen*sin((*xyzTx)[ij].dip)/2.0 );
			(*xyzTx)[ij].izb = searchInsert( MG.Z, K, (*xyzTx)[ij].z + (*xyzTx)[ij].BipoLen*sin((*xyzTx)[ij].dip)/2.0 );
			if ((*xyzTx)[ij].izb > (*xyzTx)[ij].iza )
					(*xyzTx)[ij].izb--;
			else if ((*xyzTx)[ij].izb < (*xyzTx)[ij].iza )
					(*xyzTx)[ij].iza--;
			else
			;	
			(*xyzTx)[ij].iza1 = searchInsert( MG.Z, K, (*xyzTx)[ij].z1 - (*xyzTx)[ij].BipoLen1*sin((*xyzTx)[ij].dip1)/2.0 );
			(*xyzTx)[ij].izb1 = searchInsert( MG.Z, K, (*xyzTx)[ij].z1 + (*xyzTx)[ij].BipoLen1*sin((*xyzTx)[ij].dip1)/2.0 );	
			if ((*xyzTx)[ij].izb1 > (*xyzTx)[ij].iza1 )
					(*xyzTx)[ij].izb1--;
			else if ((*xyzTx)[ij].izb1 < (*xyzTx)[ij].iza1 )
					(*xyzTx)[ij].iza1--;
			else
			;	
			(*xyzTx)[ij].ictx=searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x); 
			(*xyzTx)[ij].icty=searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y);
			(*xyzTx)[ij].ictz=searchInsert(MG.Z, K, (*xyzTx)[ij].z);		
			(*xyzTx)[ij].ictx1=searchInsert(MG.X_Bzold, L, (*xyzTx)[ij].x1); 
			(*xyzTx)[ij].icty1=searchInsert(MG.Y_Bzold, M, (*xyzTx)[ij].y1);
			(*xyzTx)[ij].ictz1=searchInsert(MG.Z, K, (*xyzTx)[ij].z1);
		} 
		else
		{ 
			printf("Error: The orentation for transmitter #%d is illegal, currently only support x,y,or z oriented bipole\n",ij+1);
			exit(-1);
		}
	}
	
	//-------------------------------------------------------------------------------------------------------------------------------------------
	tRx->Times=(double *)malloc((tRx->Nt)*sizeof(double));
	if ( (tRx->Times) == NULL )
	{
		printf("Faile to allocate memory for recording time\n");
		perror("Error: ");
		return(-1);
	}
	if( tRx->IsLog )
	{
		logspace(tRx->Times, tRx->T0, tRx->Tn, tRx->Nt);
		printf("%5d time sampling points from %f to %f in log scale\n",tRx->Nt,tRx->T0,tRx->Tn);
	}
	else
	{
		linspace(tRx->Times, tRx->T0, tRx->Tn, tRx->Nt);
		printf("%5d time sampling points from %f to %f in linear scale\n",tRx->Nt,tRx->T0,tRx->Tn);
	}	
	tRx->NComp=0;
	for(ij=0;ij<6;ij++)
	{
		if( tRx->isComp[ij] )
			tRx->NComp++;
	}	
	
	tRx->nefcmp=0;
	for(ij=0;ij<3;ij++)
	{
		if( tRx->isComp[ij] )
			tRx->nefcmp++;
	}	
	TRACE("Recording time parameters were loaded successfully\n");
	return 0;	
}

int FreeRxTxArray(RxTime *tRx, RxPos *xyzRx, TxPos **xyzTx, int NRx, int NTx)
{
	int ij;
	free(xyzRx->x);
	free(xyzRx->y);
	free(xyzRx->z);
	free(xyzRx->ix);
	free(xyzRx->iy);
	free(xyzRx->iz);
		
	free(tRx->Times);
	for(ij=0;ij<NTx;ij++)
		free((*xyzTx)[ij].RxNum);
	free(*xyzTx);	
}
