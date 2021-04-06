#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Spline.h"

double Spline3D4Ps(double X[4], double Y[4], double Z[4], double data[4][4][4], double x, double y, double z)
{
	int i,j,k;
	std::vector<double> sX(4), sY(4);
	double term[4][4],array[4];
	tk::spline s;
	//printf("x1=%f,x2=%f,x3=%f,x4=%f\n",X[0],X[1],X[2],X[3]);
	//printf("y1=%f,y2=%f,y3=%f,y4=%f\n",Y[0],Y[1],Y[2],Y[3]);
	//printf("z1=%f,z2=%f,z3=%f,z4=%f\n",Z[0],Z[1],Z[2],Z[3]);
	
	//X direction interpolation
	for(i=0;i<4;i++)
		sX[i]=X[i];
	
	for(k=0;k<4;k++)
	{
		for(j=0;j<4;j++)
		{
		 	for(i=0;i<4;i++)
		 	{
		 		sY[i]=data[k][j][i];
		 	}	
		 	s.set_points(sX,sY);
		 	term[k][j]=s(x);
		 } 
	}
	
	//Y dir interpolation
	for(j=0;j<4;j++)
		sX[j]=Y[j];
		
	for(k=0;k<4;k++)
	{
		for(j=0;j<4;j++)
			sY[j]=term[k][j];
			
		s.set_points(sX,sY);
		array[k]=s(y);	
	}		 
	
	//Z dir interpolation
	for(k=0;k<4;k++)
	{
		sX[k]=Z[k];	 
		sY[k]=array[k];
	}	 	
	s.set_points(sX,sY);
	
	return s(z);
}

