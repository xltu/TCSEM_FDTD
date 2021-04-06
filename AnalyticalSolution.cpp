#include"Hankel.h"
#include<math.h>
#include"specialfunctions.h"
#include"FTDT.h"
#define Integral_node 300
extern double miu;
double Fuc(double z,double r,double t)
{
  return z/(2*sqrt(PI*t*t*t))*exp(-r*r*t-z*z/(4*t));
}

double XKernal(double r,double z,double t,double sita)
{
  double pt=t/(miu*sita);
  double term=z/(2*sqrt(pt))+r*sqrt(pt);
  if (term<-4)
    return r*(exp(-z*z/(4*pt)-r*r*pt)/(sqrt(PI*pt))-r*exp(r*z)*erfc(term));
  else
    return r*(exp(-z*z/(4*pt)-r*r*pt)/(sqrt(PI*pt))-r*exp(r*z-term*term)*DERFCX(term));
}

double YKernal(double r,double z,double t,double sita)
{
  double pt0,pt=t/(miu*sita);
  double dt,Integral;
  int i;
  pt0=z*z/(3+sqrt(4*z*z*r*r+9));
  if(pt0>=pt)
  {
    dt=pt/Integral_node;
    Integral=0.;
    for(i=1;i<Integral_node;i++)
    {
      Integral+=dt/6.*(Fuc(z,r,i*dt)+4*Fuc(z,r,(i+0.5)*dt)+Fuc(z,r,(i+1)*dt));
    }
  }
  else
  {
    dt=pt0/Integral_node;
    Integral=0.;
    for(i=1;i<Integral_node;i++)
    {
      Integral+=dt/6.*(Fuc(z,r,i*dt)+4*Fuc(z,r,(i+0.5)*dt)+Fuc(z,r,(i+1)*dt));
    }
    dt=(pt-pt0)/100;
    for(i=0;i<100;i++)
    {
      Integral+=dt/6.*(Fuc(z,r,i*dt+pt0)+4*Fuc(z,r,(i+0.5)*dt+pt0)+Fuc(z,r,(i+1)*dt+pt0));
    }
  }
    return Integral;
}
double DXKernal(double r,double z,double t,double sita)
{
  double pt=t/(miu*sita);
  if(z==0)
  {
    return -r/(miu*sita*sqrt(PI*pt*pt*pt))*exp(-r*r*pt);
  }
  else
  {
    return r/(miu*sita*z)*Fuc(z,r,pt)*(sqrt(pow(alglib::besseljn(2,0.5*z/sqrt(pt)),2)+
                                             pow(alglib::besselyn(2,0.5*z/sqrt(pt)),2))-2*r*z);
  }
}
double DYKernal(double r,double z,double t,double sita)
{
  double pt=t/(miu*sita);
  return Fuc(z,r,pt);
}
double AnalyticalSolution(int tag,double t,int xint,int yint,int zint,double sita,ModelPara *MP,ModelGrid *MG)
{

  double x,y,z,rou,E,term,term1;
  double YB[801];
  int i,j,k;
  if(tag==0)
  {
    ///Ex magnetic dipole step response
    x=MG->X_Bzold[xint]-(MG->X_Bzold[MP->AnLx_old+(MP->Nx-1)/2]);

    if(yint<2*MP->Any_old+MP->Ny)
      y=MG->Y_Bzold[yint]-0.5*MG->dy[yint]-(MG->Y_Bzold[MP->Any_old+(MP->Ny-1)/2]);
    else
      y=MG->Y_Bzold[yint-1]+0.5*MG->dy[yint-1]-(MG->Y_Bzold[MP->Any_old+(MP->Ny-1)/2]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];

    rou=sqrt(x*x+y*y);
    if(rou==0)
        rou=0.01;
    for(i=0;i<801;i++)
    {
      term=YBASE[i]/rou;
      YB[i]=term*XKernal(term,z,t,sita);
    }

    E=0;
    for(i=0;i<801;i++)
        E+=YB[i]*WT1[i];

    return y*E/(rou*rou)/(2*PI*sita);
  }
  else if(tag==1)
  {
    ///Ey magnetic dipole step-response
    if(xint<MP->AnLx_old+MP->AnRx_old+MP->Nx)
      x=MG->X_Bzold[xint]-0.5*MG->dx[xint]-(MG->X_Bzold[MP->AnLx_old+(MP->Nx-1)/2]);
    else
      x=MG->X_Bzold[xint-1]+0.5*MG->dx[xint-1]-(MG->X_Bzold[MP->AnLx_old+(MP->Nx-1)/2]);

    y=MG->Y_Bzold[yint]-(MG->Y_Bzold[MP->Any_old+(MP->Ny-1)/2]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];

    rou=sqrt(x*x+y*y);
    if(rou==0)
        rou=0.01;
    for(i=0;i<801;i++)
    {
      term=YBASE[i]/rou;
      YB[i]=term*XKernal(term,z,t,sita);
    }

    E=0;
    for(i=0;i<801;i++)

        E+=YB[i]*WT1[i];

    return -x*E/(rou*rou)/(2*PI*sita);

  }
  else if(tag==2)
  {
    /// LOTEM step-Ex
    //Needs a integral along the source line
    if(yint<2*MP->Any_old+MP->Ny)
      y=MG->Y_Bzold[yint]-0.5*MG->dy[yint]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);
    else
      y=MG->Y_Bzold[yint-1]+0.5*MG->dy[yint-1]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;
    E=0.;
    for(i=MP->ScXa;i<MP->ScXb;i++)
    {
      x=MG->X_Bzold[xint]-MG->X_Bzold[i];
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*XKernal(term,z,t,sita);
      }

      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT0[j];
      term1/=rou;

      E+=(MG->dx[i]*term1);
    }
    for(i=0;i<2;i++)
    {
      if(i==0)
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*YKernal(term,z,t,sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*x/rou*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==3)
  {
    ///LOTEM step-Ey
    //integral along the source wire
    y=MG->Y_Bzold[yint]-(MG->Y_Bzold[MP->ScYa]);
    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;

    E=0.;
    for(i=0;i<2;i++)
    {
      if(xint<MP->AnLx_old+MP->AnRx_old+MP->Nx)
        x=MG->X_Bzold[xint]-0.5*MG->dx[xint];
      else
        x=MG->X_Bzold[xint-1]+0.5*MG->dx[xint-1];

      if(i==0)
      {
        x=x-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=x-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*YKernal(term,z,t,sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*y/rou*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==4)
  {
    /// Ex LOTEM impulse response
    //Needs a integral along the source line
    if(yint<2*MP->Any_old+MP->Ny)
      y=MG->Y_Bzold[yint]-0.5*MG->dy[yint]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);
    else
      y=MG->Y_Bzold[yint-1]+0.5*MG->dy[yint-1]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;
    E=0.;
    for(i=MP->ScXa;i<MP->ScXb;i++)
    {
      x=MG->X_Bzold[xint]-MG->X_Bzold[i];
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=DXKernal(term,z,t,sita);
      }

      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT0[j];
      term1/=rou;

      E+=(MG->dx[i]*term1);
    }
    for(i=0;i<2;i++)
    {
      if(i==0)
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*DYKernal(term,z,t,sita)/(sita*miu);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*x/rou*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==5)
  {
    /// Ey LOTEM impulse response
    //integral along the source wire
    y=MG->Y_Bzold[yint]-(MG->Y_Bzold[MP->ScYa]);
    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;

    E=0.;
    for(i=0;i<2;i++)
    {
      if(xint<MP->AnLx_old+MP->AnRx_old+MP->Nx)
        x=MG->X_Bzold[xint]-0.5*MG->dx[xint];
      else
        x=MG->X_Bzold[xint-1]+0.5*MG->dx[xint-1];

      if(i==0)
      {
        x=x-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=x-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*DYKernal(term,z,t,sita)/(miu*sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*y/rou*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==6)
  {
    /// Ez LOTEM impulse response
    //integral along the source wire
    y=MG->Y_Bzold[yint]-(MG->Y_Bzold[MP->ScYa]);
    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;

    E=0.;
    for(i=0;i<2;i++)
    {
      if(xint<MP->AnLx_old+MP->AnRx_old+MP->Nx)
        x=MG->X_Bzold[xint]-0.5*MG->dx[xint];
      else
        x=MG->X_Bzold[xint-1]+0.5*MG->dx[xint-1];

      if(i==0)
      {
        x=x-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=x-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*DYKernal(term,z,t,sita)/(miu*sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT0[j];
      term1/=rou;
      E+=(pow(-1,i)*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==7)
  {
    /// Bx LOTEM impulse response
    //integral along the source wire
    y=MG->Y_Bzold[yint]-(MG->Y_Bzold[MP->ScYa]);
    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;

    E=0.;
    for(i=0;i<2;i++)
    {
      if(xint<MP->AnLx_old+MP->AnRx_old+MP->Nx)
        x=MG->X_Bzold[xint]-0.5*MG->dx[xint];
      else
        x=MG->X_Bzold[xint-1]+0.5*MG->dx[xint-1];

      if(i==0)
      {
        x=x-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=x-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*XKernal(term,z,t,sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*y/rou*term1);
    }
    return E*Current/(4*PI*sita);
  }
  else if(tag==8)
  {
    /// By LOTEM impulse response
    //Needs a integral along the source line
    if(yint<2*MP->Any_old+MP->Ny)
      y=MG->Y_Bzold[yint]-0.5*MG->dy[yint]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);
    else
      y=MG->Y_Bzold[yint-1]+0.5*MG->dy[yint-1]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;
    E=0.;
    for(i=MP->ScXa;i<MP->ScXb;i++)
    {
      x=MG->X_Bzold[xint]-MG->X_Bzold[i];
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=DYKernal(term,z,t,sita)-2*term*XKernal(term,z,t,sita);
      }

      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT0[j];
      term1/=rou;

      E+=(MG->dx[i]*term1);
    }
    for(i=0;i<2;i++)
    {
      if(i==0)
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXa];
      }
      else
      {
        x=MG->X_Bzold[xint]-MG->X_Bzold[MP->ScXb-1];
      }
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*XKernal(term,z,t,sita);
      }
      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;
      E+=(pow(-1,i)*x/rou*term1);
    }
    return -E*Current/(4*PI*sita);
  }
  else if(tag==9)
  {
    /// Bz LOTEM impulse response
    //Needs a integral along the source line
    if(yint<2*MP->Any_old+MP->Ny)
      y=MG->Y_Bzold[yint]-0.5*MG->dy[yint]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);
    else
      y=MG->Y_Bzold[yint-1]+0.5*MG->dy[yint-1]-(MG->Y_Bzold[MP->ScYa]-0.5*MG->dy[MP->ScYa]);

    z=0;
    for(k=1;k<=zint;k++)
      z+=MG->dz[k-1];
    if(z==0)
      z=0.1;
    E=0.;
    for(i=MP->ScXa;i<MP->ScXb;i++)
    {
      x=MG->X_Bzold[xint]-MG->X_Bzold[i];
      rou=sqrt(x*x+y*y);
      if(rou==0)
        rou=0.01;
      for(j=0;j<801;j++)
      {
        term=YBASE[j]/rou;
        YB[j]=2*term*XKernal(term,z,t,sita);
      }

      term1=0.;
      for(j=0;j<801;j++)
        term1+=YB[j]*WT1[j];
      term1/=rou;

      E+=(MG->dx[i]*y/rou*term1);
    }
    return E*Current/(4*PI*sita);
  }
  else
    return 0.;
}
