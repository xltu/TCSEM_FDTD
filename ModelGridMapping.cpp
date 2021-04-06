#include<math.h>
#include<omp.h>
#include"FTDT.h"
/*
	function ModelGridMapping maps the grid position between the uniform grid and 
	the irregular grid (grid size increase exponentially).
	The results of it mapping is returned in the two structure pointer ModelGrid
	and GridConv.
	
	@Xiaolei Tu
	@tuxl2009@hotmail.com    
*/
int ModelGridMapping(ModelPara *MP,ModelGrid *MG,GridConv *GC)
/* AnLx  the left irregular perism numbers
 * AnRx  the right irregular perism numbers
 * Nx	  the regular perism numbers in the middle
 * Any   the irregular perism numbers in front and back of the model
 * Ny    the regular persim numbers in the y direction
 * AnLx_new  the regular perism numbers interpolated into the left
 * AnRx_new  the regular persim numbers interpolated into the right domain
 * Any_new   the regular persim bumbers interpolated into the y direction in the irregular segments

 * MG->dx[AnLx+AnRx+Nx]  the x temporal steps
 * MG->dy[2*Any+Ny]		the y temporal steps
 * MG->X_BzNew[AnLx_new+AnRx_new+Nx] the X coordinates for the interpolated Bz in the new grid
 * MG->Y_BzNew[2MP->Any_New+Ny] the Y coordinates for the interpolated Bz in the new grid
 * MG->X_Bzold[AnLx+AnRx+Nx] the X coordinates for the orginal Bz in the old grid
 * MG->Y_Bzold[2*Any+Ny] 		the Y coordinates for the orginal Bz in tge old grid
 * GC->xBzNew2old[AnLx_new+AnRx_New+Nx] the search index for old segments, that is MG->X_BzNew[i] loacated between MG->X_Bzold[GC->xBzNew2old[i]]
 												and MG->X_Bzold[GC->xBzNew2old[i]+1]
 * GC->yBzNew2old[2MP->Any_New+Ny]			the same as above
 * GC->xBzold2New[AnLx+AnRx+Nx]       the search index for new segments, that is MG->X_Bzold[i] located between MG->X_BzNew[GC->xBzold2New[i]]
 												and MG->X_BzNew[GC->xBzold2New[i]+1]
 * GC->yBzold2New[2*Any+Ny]           above
 * GC->xoldBx2NewBz[L+1]					the new Bx have the same coordinates with newBz, so to loacated old Bx into its new grid is 													the same as loacted old Bz into newBx, that is a transfor from old index to new index
 * GC->yoldBy2NewBz[M+1]   				see above
 */

{
	TRACE("Creating mapping between irregular spaced mesh and regular spaced mesh\n");
	int i,j;
	int L0,M0,L1,M1;
  int AnRx,AnLx,Nx,Any,Ny;
	
	AnRx=MP->ARx1+MP->ARx2;
	AnLx=MP->ALx1+MP->ALx2;
	Nx=MP->Nx;
	Any=MP->Ay1+MP->Ay2;
	Ny=MP->Ny;

	L0=MP->L;
	M0=MP->M;

  L1=MP->AnLx_New+MP->AnRx_New+Nx;
	M1=2*(MP->Any_New)+Ny;

	
	///caculate the new coordinates
	//#pragma omp parallel for  if (Nx>200)
	for(i=MP->AnLx_New;i<MP->AnLx_New+Nx;i++)
		MG->X_BzNew[i]=MG->X_Bzold[i-MP->AnLx_New+AnLx];
		
	for(i=MP->AnLx_New-1;i>=0;i--)
		MG->X_BzNew[i]=MG->X_BzNew[i+1]-MP->dxmin;
	for(i=MP->AnLx_New+Nx;i<L1;i++)
		MG->X_BzNew[i]=MG->X_BzNew[i-1]+MP->dxmin;
		
	//#pragma omp parallel for  if (Ny>200)
	for(j=MP->Any_New;j<MP->Any_New+Ny;j++)
		MG->Y_BzNew[j]=MG->Y_Bzold[j-MP->Any_New+Any];
		
	for(j=MP->Any_New-1;j>=0;j--)
		MG->Y_BzNew[j]=MG->Y_BzNew[j+1]-MP->dymin;
	for(j=MP->Any_New+Ny;j<M1;j++)
		MG->Y_BzNew[j]=MG->Y_BzNew[j-1]+MP->dymin;

	///locate Bz sampling notes from new grid into old grid
	/* MG->X_BzNew[i] lies in the left of MG->X_Bzold[0]      												GC->xBzNew2old[i]=0;
	 * MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]      	0<=i<MP->AnLx_New
	 * MG->X_BzNew[i] lies at the same point with MG->X_Bzold[GC->xBzNew2old[i]]     				 		MP->AnLx_New<=i<MP->AnLx_New+Nx
	 * MG->X_BzNew[i] lies between MG->X_Bzold[GC->xBzNew2old[i]-1] and MG->X_Bzold[GC->xBzNew2old[i]]			MP->AnLx_New+Nx<=i<MP->AnLx_New+MP->AnRx_New+Nx
	 * MG->X_BzNew[i] lies in the right of MG->X_Bzold[L-1]                                       GC->xBzNew2old[i]=L;
	*/
	//#pragma omp parallel for if ( Nx>200 ) 
	for(i=MP->AnLx_New;i<MP->AnLx_New+Nx;i++)
		GC->xBzNew2old[i]=i-MP->AnLx_New+AnLx;
		
	for(i=MP->AnLx_New-1;i>=0;i--)
	{
		j=GC->xBzNew2old[i+1];
		while(j>0 && MG->X_Bzold[j-1]>=MG->X_BzNew[i])
			j--;
		GC->xBzNew2old[i]=j;
	}

	for(i=MP->AnLx_New+Nx;i<L1;i++)
	{
		j=GC->xBzNew2old[i-1]-1;
		while(j<L0-1 && MG->X_Bzold[j+1]<MG->X_BzNew[i])
			j++;
		GC->xBzNew2old[i]=j+1;
	}
  //-------------------------------------------------------------------------------------------------------------
#if DEBUG == 1   
  ///test code below
  #pragma omp parallel for  if ( L1>200 ) 
  for(i=0;i<L1;i++)
  {
    if(GC->xBzNew2old[i]==0)
    {
      if(MG->X_BzNew[i]>MG->X_Bzold[0])
        printf("error in xBzNew2old,%d\n",i);
    }
    else if(GC->xBzNew2old[i]==L0)
    {
      if(MG->X_BzNew[i]<=MG->X_Bzold[L0-1])
        printf("error in xBzNew2old,%d\n",i);
    }
    else
    {
      if(MG->X_BzNew[i]<=MG->X_Bzold[GC->xBzNew2old[i]-1] || MG->X_BzNew[i]>MG->X_Bzold[GC->xBzNew2old[i]])
        printf("error in xBzNew2old,%d\n",i);
    }
  }
  ///test code above
#endif  
  //-------------------------------------------------------------------------------------------------------------
	/* MG->Y_BzNew[j] lies in the front of MG->Y_Bzold[0]                         					GC->yBzNew2old[i]=0;
	 * MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]      	0<=i<MP->Any_New
	 * MG->Y_BzNew[j] lies at the same point as MG->Y_Bzold[GC->yBzNew2old[j]]              			MP->Any_New<=i<MP->Any_New+Ny
	 * MG->Y_BzNew[j] lies between MG->Y_Bzold[GC->yBzNew2old[j]-1] and MG->Y_Bzold[GC->yBzNew2old[j]]			MP->Any_New+Ny<=i<M1
	 * MG->Y_BzNew[j] lies in the back of MG->Y_Bzold[M-1]													 GC->yBzNew2old[i]=M;
	*/
	//#pragma omp parallel for  if (Ny>200)
	for(j=MP->Any_New;j<MP->Any_New+Ny;j++)
		GC->yBzNew2old[j]=j-MP->Any_New+Any;
		
	for(j=MP->Any_New-1;j>=0;j--)
	{
		i=GC->yBzNew2old[j+1];
		while(i>0 && MG->Y_Bzold[i-1]>=MG->Y_BzNew[j])
			i--;
		GC->yBzNew2old[j]=i;
	}
	for(j=MP->Any_New+Ny;j<M1;j++)
	{
		i=GC->yBzNew2old[j-1]-1;
		while(i<M0-1 && MG->Y_Bzold[i+1]<MG->Y_BzNew[j])
			i++;
		GC->yBzNew2old[j]=i+1;
	}
#if DEBUG == 1 
  ///test code below
  #pragma omp parallel for  if ( M1>200 ) 
  for(i=0;i<M1;i++)
  {
    if(GC->yBzNew2old[i]==0)
    {
      if(MG->Y_BzNew[i]>MG->Y_Bzold[0])
        printf("error in yBzNew2old,%d\n",i);
    }
    else if(GC->yBzNew2old[i]==M0)
    {
      if(MG->Y_BzNew[i]<=MG->Y_Bzold[M0-1])
        printf("error in yBzNew2old,%d\n",i);
    }
    else
    {
      if(MG->Y_BzNew[i]<=MG->Y_Bzold[GC->yBzNew2old[i]-1] || MG->Y_BzNew[i]>MG->Y_Bzold[GC->yBzNew2old[i]])
        printf("error in yBzNew2old,%d\n",i);
    }
  }
  ///test code above
#endif
	///locate Bz sampling notes from old grid into new grid
	/* MG->X_Bzold[i] lies in the left of MG->X_BzNew[0] 											GC->xBzold2New[i]=0
	 * MG->X_Bzold[i] lies between MG->X_BzNew[GC->xBzold2New[i]-1] and MG->X_BzNew[GC->xBzold2New[i]]; 0<=i<AnLx
	 * MG->X_Bzold[i] lies at the same point with MG->X_BzNew[GC->xBzold2New[i]]                AnLx<=i<AnLx+Nx
	 * MG->X_Bzold[i] lies between MG->X_BzNew[GC->xBzold2New[i]-1] and MG->X_BzNew[GC->xBzold2New[i]]  AnLx+Nx<=i<L
	 * MG->X_Bzold[i] lies in the right of MG->X_BzNew[L1-1]                              GC->xBzold2New[i]=L1
	*/
	//#pragma omp parallel for  if ( Nx>200 ) 
	for(i=AnLx;i<AnLx+Nx;i++)
		GC->xBzold2New[i]=i-AnLx+MP->AnLx_New;
	for(i=AnLx-1;i>=0;i--)
	{
		j=GC->xBzold2New[i+1];
		while(j>0 && MG->X_BzNew[j-1]>=MG->X_Bzold[i])
			j--;
		GC->xBzold2New[i]=j;
	}
	for(i=AnLx+Nx;i<L0;i++)
	{
		j=GC->xBzold2New[i-1];
		while(j<L1-1 && MG->X_BzNew[j+1]<MG->X_Bzold[i])
			j++;
		GC->xBzold2New[i]=j+1;
	}
#if DEBUG == 1 
  ///test code below
  #pragma omp parallel for  if ( L0>200 ) 
  for(i=0;i<L0;i++)
  {
    if(GC->xBzold2New[i]==0)
    {
      if(MG->X_Bzold[i]>MG->X_BzNew[0])
        printf("error in xBzold2New,%d\n",i);
    }
    else if(GC->xBzold2New[i]==L1)
    {
      if(MG->X_Bzold[i]<=MG->X_BzNew[L1-1])
        printf("error in xBzold2New,%d\n",i);
    }
    else
    {
      if(MG->X_Bzold[i]<=MG->X_BzNew[GC->xBzold2New[i]-1] || MG->X_Bzold[i]>MG->X_BzNew[GC->xBzold2New[i]])
        printf("error in xBzold2New,%d\n",i);
    }
  }
  ///test code above
#endif
	/* MG->Y_Bzold[j] lies in the front of MG->Y_BzNew[0]                                        GC->yBzold2New[j]=0
	 * MG->Y_Bzold[j] lies between      MG->Y_BzNew[GC->yBzold2New[j]-1] and MG->Y_BzNew[GC->yBzold2New[j]]; 0<=j<Any
	 * MG->Y_Bzold[j] lies at the point MG->Y_BzNew[GC->yBzold2New[j]]                                Any<=j<Any+Ny
	 * MG->Y_Bzold[j] lies between      MG->Y_BzNew[GC->yBzold2New[j]-1] and MG->Y_BzNew[GC->yBzold2New[j]]  Any+Ny<=j<M
	 * MG->Y_Bzold[j] lies in the back of MG->Y_BzNew[M1-1]                                        GC->yBzold2New[j]=M1
	 */
	// #pragma omp parallel for  if (Ny>200)
	for(j=Any;j<Any+Ny;j++)
		GC->yBzold2New[j]=j-Any+MP->Any_New;
	for(j=Any-1;j>=0;j--)
	{
		i=GC->yBzold2New[j+1];
		while(i>0 && MG->Y_BzNew[i-1]>=MG->Y_Bzold[j])
			i--;
		GC->yBzold2New[j]=i;
	}
	for(j=Any+Ny;j<M0;j++)
	{
		i=GC->yBzold2New[j-1];
		while(i<M1-1 && MG->Y_BzNew[i+1]<MG->Y_Bzold[j])
			i++;
		GC->yBzold2New[j]=i+1;
	}
#if DEBUG == 1 
  ///test code below
  #pragma omp parallel for  if ( ( M0>200 )  ) 
  for(i=0;i<M0;i++)
  {
    if(GC->yBzold2New[i]==0)
    {
      if(MG->Y_Bzold[i]>MG->Y_BzNew[0])
        printf("error in yBzold2New,%d\n",i);
    }
    else if(GC->yBzold2New[i]==M1)
    {
      if(MG->Y_Bzold[i]<=MG->Y_BzNew[M1-1])
        printf("error in yBzold2New,%d\n",i);
    }
    else
    {
      if(MG->Y_Bzold[i]<=MG->Y_BzNew[GC->yBzold2New[i]-1] || MG->Y_Bzold[i]>MG->Y_BzNew[GC->yBzold2New[i]])
        printf("error in yBzold2New,%d\n",i);
    }
  }
  ///test code above
#endif
	/* locate Bx sampling notes from old grid into new grid in the x direction, the y direction is tha same as GC->yBzold2New
	 * X_Bxold[i]=MG->X_Bzold[i]-0.5*dx[i] i=[0,L-1]
	 * X_Bxold[L]=MG->X_Bzold[L-1]+0.5*dx[L-1];
	 * MG->Y_Bzold[j]=MG->Y_Bzold[j]  j=[0;M-1];
	 * X_BxNew[i]=MG->X_BzNew[i]   i=[0,L1-1]
	 * Y_BxNew[j]=MG->Y_BzNew[j]		j=[0,M1-1]

	 * X_Bxold[i] lies in the left of X_BxNew[0]													  GC->xoldBx2NewBz[i]=0;
	 * X_Bxold[i] lies between X_BxNew[GC->xoldBx2NewBz[i]-1] and X_BxNew[GC->xoldBx2NewBz[i]]  0<=i<AnLx
	 * X_Bxold[i] lies between X_BxNew[GC->xoldBx2NewBz[i]-1] and X_BxNew[GC->xoldBx2NewBz[i]]  AnLx<=i<AnLx+Nx
	 * X_Bxold[i] lies between X_BxNew[GC->xoldBx2NewBz[i]-1] and X_BxNew[GC->xoldBx2NewBz[i]]  AnLx+Nx<=i<=AnLx+AnRx+Nx
	 * X_Bxold[i] lies in the right of X_BxNew[L1-1]                                    GC->xoldBx2NewBz[i]=L1;
	*/
	//#pragma omp parallel for  if ( Nx>200 ) 
	for(i=AnLx;i<AnLx+Nx;i++)
		GC->xoldBx2NewBz[i]=i-AnLx+MP->AnLx_New;
	for(i=AnLx-1;i>=0;i--)
	{
		j=GC->xoldBx2NewBz[i+1];
		while(j>0 && MG->X_BzNew[j-1]>=MG->X_Bzold[i]-0.5*MG->dx[i])
			j--;
		GC->xoldBx2NewBz[i]=j;
	}
	for(i=AnLx+Nx;i<L0;i++)
	{
		j=GC->xoldBx2NewBz[i-1];
		while(j<L1-1 && MG->X_BzNew[j+1]<MG->X_Bzold[i]-0.5*MG->dx[i])
			j++;
		GC->xoldBx2NewBz[i]=j+1;
	}
	i=L0;
	{
		j=GC->xoldBx2NewBz[i-1];
		while(j<L1-1 && MG->X_BzNew[j+1]<MG->X_Bzold[i-1]+0.5*MG->dx[i-1])
			j++;
		GC->xoldBx2NewBz[i]=j+1;
	}
#if DEBUG == 1 
  ///test code below
  #pragma omp parallel for  if ( L0>200 ) 
  for(i=0;i<L0;i++)
  {
    if(GC->xoldBx2NewBz[i]==0)
    {
      if(MG->X_Bzold[i]-0.5*MG->dx[i]>MG->X_BzNew[0])
        printf("error in xoldBx2NewBz,%d\n",i);
    }
    else if(GC->xoldBx2NewBz[i]==L1)
    {
      if(MG->X_Bzold[i]-0.5*MG->dx[i]<=MG->X_BzNew[L1-1])
        printf("error in xoldBx2NewBz,%d\n",i);
    }
    else
    {
      if(MG->X_Bzold[i]-0.5*MG->dx[i]<=MG->X_BzNew[GC->xoldBx2NewBz[i]-1] ||
          MG->X_Bzold[i]-0.5*MG->dx[i]>MG->X_BzNew[GC->xoldBx2NewBz[i]])
        printf("error in xoldBx2NewBz,%d\n",i);
    }
  }
  ///test code above
#endif
	///locate By sampling notes from old grid to new grid in the y direction
	///the locating mapping in the x direction is the same as GC->xBzold2New
	/* X_Byold[i]=MG->X_Bzold[i]   					i=[0,L-1]
	 * Y_Byold[j]=MG->Y_Bzold[j]-0.5*dy[j] 		j=[0,M-1]
	 * Y_Byold[L]=MG->Y_Bzold[L-1]+0.5*dy[L-1]
	 * X_ByNew[i]=MG->X_BzNew[i]    				i=[0,L1-1]
	 * Y_ByNew[j]=MG->Y_BzNew[j]						j=[0,M1-1]
	 *
	 * Y_Byold[j] lies in the front of Y_ByNew[0]													  GC->yoldBy2NewBz[j]=0;
	 * Y_Byold[j] lies between 	Y_ByNew[GC->yoldBy2NewBz[j]-1] and Y_ByNew[GC->yoldBy2NewBz[j]]  0<=j<Any
	 * Y_Byold[j] lies between  Y_ByNew[GC->yoldBy2NewBz[j]-1] and Y_ByNew[GC->yoldBy2NewBz[j]]  Any<=i<Any+Ny
	 * Y_Byold[j] lies between  Y_ByNew[GC->yoldBy2NewBz[j]-1] and Y_ByNew[GC->yoldBy2NewBz[j]] Any+Ny<=i<=M
	 * Y_Byold[j] lies in the back of Y_ByNew[M1-1]                                    GC->yoldBy2NewBz[j]=M1;
	*/
	//#pragma omp parallel for  if (Ny>200)
	for(j=Any;j<Any+Ny;j++)
		GC->yoldBy2NewBz[j]=j-Any+MP->Any_New;
	for(j=Any-1;j>=0;j--)
	{
		i=GC->yoldBy2NewBz[j+1];
		while(i>0 && MG->Y_BzNew[i-1]>=MG->Y_Bzold[j]-0.5*MG->dy[j])
			i--;
		GC->yoldBy2NewBz[j]=i;
	}
	for(j=Any+Ny;j<M0;j++)
	{
		i=GC->yoldBy2NewBz[j-1];
		while(i<M1-1 && MG->Y_BzNew[i+1]<MG->Y_Bzold[j]-0.5*MG->dy[j])
			i++;
		GC->yoldBy2NewBz[j]=i+1;
	}
	j=M0;
	{
		i=GC->yoldBy2NewBz[j-1];
		while(i<M1-1 && MG->Y_BzNew[i+1]<MG->Y_Bzold[j-1]+0.5*MG->dy[j-1])
			i++;
		GC->yoldBy2NewBz[j]=i+1;
	}

  ///test code below
#if DEBUG == 1  
  #pragma omp parallel for  if ( ( M0>200 )  ) 
  for(i=0;i<M0;i++)
  {
    if(GC->yoldBy2NewBz[i]==0)
    {
      if(MG->Y_Bzold[i]-0.5*MG->dy[i]>MG->Y_BzNew[0])
        printf("error in yoldBy2NewBz,%d\n",i);
    }
    else if(GC->yoldBy2NewBz[i]==M1)
    {
      if(MG->Y_Bzold[i]-0.5*MG->dy[i]<=MG->Y_BzNew[M1-1])
        printf("error in yoldBy2NewBz,%d\n",i);
    }
    else
    {
      if(MG->Y_Bzold[i]-0.5*MG->dy[i]<=MG->Y_BzNew[GC->yoldBy2NewBz[i]-1] ||
          MG->Y_Bzold[i]-0.5*MG->dy[i]>MG->Y_BzNew[GC->yoldBy2NewBz[i]])
        printf("error in yoldBy2NewBz,%d\n",i);
    }
  }
#endif
  ///test code above
	TRACE("Mesh mapping was created successfully\n");
	
	return 0;
}
