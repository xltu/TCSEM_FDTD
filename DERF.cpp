#include<stdio.h>
#include<math.h>
double CALERF(double X,int JINT)
{
	double SQRPI=5.6418958354775628695e-1;
	double THRESH=0.468750;
	double SIXTEN=16.00;
	double FOUR=4.00;
	double ONE=1.00;
	double HALF=0.50;
	double TWO=2.00;
	double ZERO=0.00;
	double XINF=1.79e308;
	double XNEG=-26.6280;
	double XSMALL=1.11e-16;
	double XBIG=26.5430;
	double XHUGE=6.71e7;
	double XMAX=2.53e307;
	double A[5]={3.16112374387056560e00,1.13864154151050156e02,3.77485237685302021e02,3.20937758913846947e03,1.85777706184603153e-1};
	double B[4]={2.36012909523441209e01,2.44024637934444173e02,1.28261652607737228e03,2.84423683343917062e03};
	double C[9]={5.64188496988670089e-1,8.88314979438837594e0,6.61191906371416295e01,2.98635138197400131e02,
			       8.81952221241769090e02,1.71204761263407058e03,2.05107837782607147e03,1.23033935479799725e03,2.15311535474403846e-8};
   double D[8]={1.57449261107098347e01,1.17693950891312499e02,5.37181101862009858e02,1.62138957456669019e03,
                3.29079923573345963e03,4.36261909014324716e03,3.43936767414372164e03,1.23033935480374942e03};
   double P[6]={3.05326634961232344e-1,3.60344899949804439e-1,1.25781726111229246e-1,1.60837851487422766e-2,
                6.58749161529837803e-4,1.63153871373020978e-2};
   double Q[5]={2.56852019228982242e00,1.87295284992346047e00,5.27905102951428412e-1,6.05183413124413191e-2,2.33520497626869185e-3};
   double RESULT,Y,YSQ,XNUM,XDEN,DEL;
   int i;
   Y = fabs(X);

   if(Y <= THRESH)
   {
   	YSQ = ZERO;
   	if(Y > XSMALL)
   		YSQ = Y * Y;
   	XNUM=A[4]*YSQ;
   	XDEN=YSQ;

   	for(i=1;i<=3;i++)
   	{
   		XNUM=(XNUM+A[i-1])*YSQ;
   		XDEN=(XDEN+B[i-1])*YSQ;
   	}
   	RESULT=X*(XNUM+A[3])/(XDEN+B[3]);
   	if(JINT!=0)
   		RESULT=ONE-RESULT;
   	if(JINT==2)
   		RESULT=exp(YSQ)*RESULT;
   	return RESULT;
   }
   else if(Y <= FOUR)
   {
   	XNUM=C[8]*Y;
   	XDEN=Y;
   	for(i=1;i<=7;i++)
   	{
   		XNUM=(XNUM+C[i-1])*Y;
   		XDEN=(XDEN+D[i-1])*Y;
   	}
   	RESULT=(XNUM+C[7])/(XDEN+D[7]);
   	if(JINT!=2)
   	{
   		YSQ=(int)(Y*SIXTEN)/SIXTEN;
   		DEL=(Y-YSQ)*(Y+YSQ);
   		RESULT=exp(-YSQ*YSQ)*exp(-DEL)*RESULT;
   	}
   }
   else
   {
   	RESULT=ZERO;
   	if(Y >= XBIG)
   	{
   		if(JINT != 2 || Y >= XMAX)
   		{
   		   if(JINT ==0)
				{
					RESULT=(HALF-RESULT)+HALF;
					if(X <ZERO)
						RESULT=-RESULT;
				}
				else if(JINT==1)
				{
					if(X<ZERO)
						RESULT=TWO-RESULT;
				}
				else
				{
					if(X<ZERO)
					{
						if(X<XNEG)
						{
							RESULT=XINF;
						}
						else
						{
							YSQ=(int)(Y*SIXTEN)/SIXTEN;
							DEL=(X-YSQ)*(X+YSQ);
							Y=exp(YSQ*YSQ) * exp(DEL);
							RESULT = (Y+Y) -RESULT;
						}
					}
				}
				return RESULT;
   		}
   		if(Y >= XHUGE)
   		{
   			RESULT=SQRPI/Y;
			   if(JINT ==0)
				{
					RESULT=(HALF-RESULT)+HALF;
					if(X <ZERO)
						RESULT=-RESULT;
				}
				else if(JINT==1)
				{
					if(X<ZERO)
						RESULT=TWO-RESULT;
				}
				else
				{
					if(X<ZERO)
					{
						if(X<XNEG)
						{
							RESULT=XINF;
						}
						else
						{
							YSQ=(int)(Y*SIXTEN)/SIXTEN;
							DEL=(X-YSQ)*(X+YSQ);
							Y=exp(YSQ*YSQ)*exp(DEL);
							RESULT=(Y+Y)-RESULT;
						}
					}
				}
				return RESULT;
   		}
   	}
   	YSQ=ONE/(Y*Y);
   	XNUM=P[5]*YSQ;
   	XDEN=YSQ;
   	for(i=1;i<=4;i++)
   	{
   		XNUM=(XNUM+P[i-1])*YSQ;
   		XDEN=(XDEN+Q[i-1])*YSQ;
   	}
   	RESULT = YSQ *(XNUM+P[4])/(XDEN+Q[4]);
   	RESULT=(SQRPI-RESULT)/Y;
   	if(JINT!=2)
   	{
   		YSQ=(int)(Y*SIXTEN)/SIXTEN;
   		DEL=(Y-YSQ)*(Y+YSQ);
   		RESULT=exp(-YSQ*YSQ)*exp(-DEL)*RESULT;
   	}
   }

   if(JINT ==0)
   {
   	RESULT=(HALF-RESULT)+HALF;
   	if(X <ZERO)
   		RESULT=-RESULT;
   }
   else if(JINT==1)
   {
   	if(X<ZERO)
   		RESULT=TWO-RESULT;
   }
   else
   {
   	if(X<ZERO)
   	{
   		if(X<XNEG)
   		{
   			RESULT=XINF;
   		}
   		else
   		{
   			YSQ=(int)(Y*SIXTEN)/SIXTEN;
   			DEL=(X-YSQ)*(X+YSQ);
   			Y=exp(YSQ*YSQ)*exp(DEL);
   			RESULT=(Y+Y)-RESULT;
   		}
   	}
   }
   return RESULT;
}

double DERF(double X)
{
	return CALERF(X,0);
}
double DERFC(double X)
{
	return CALERF(X,1);
}
double DERFCX(double X)
{
	return CALERF(X,2);
}
