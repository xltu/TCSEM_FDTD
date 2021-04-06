/* the file provide soucewave and time step for the modeling
*/
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include"FTDT.h"
#include <omp.h>
int Waveform(int N_steps,double sita_t,double *dt,double *Wave)
{
  /*Waveform( ) return a array contain the time step and source wave
   for the period that source was on
   we are modeling the impulse responce
   the impulse delta function was approximated by the gauss function
   dt returns the 2*N_steps+1 time steps
   wave returns the 2*N_steps+1 waveform of the souce
   sita_t is the standard derivation for the gauss function
  */
  double t1,t0;
  double *t;
  // the 2*N_steps+1 time point, zeros is not included
  int i;

  t=(double *)malloc((2*N_steps+1)*sizeof(double));
  if(t==NULL)
  {
    printf("fail to get memory for t!\n");
    exit(EXIT_FAILURE);
    return 1;
  }
  t[0]=0;
  for(i=0;i<N_steps;i++)
  {
    t0=t[i]-3*sita_t;
    dt[i]=-sita_t*sita_t*(1-exp(-4.5))/N_steps/t0/exp(-t0*t0/(2*sita_t*sita_t));
    t[i+1]=t[i]+dt[i];
  }
  if(3*sita_t<t[N_steps])
    t[N_steps]=3*sita_t;

  for(i= N_steps+1;i<2*N_steps+1;i++)
  {
    t[i]=6*sita_t-t[2*N_steps-i];
    dt[i-1]=t[i]-t[i-1];
  }
	
  #pragma omp parallel for  if (N_steps>150) private(i)
  for(i=0;i<2*N_steps+1;i++)
    Wave[i]=1/(sqrt(2*PI)*sita_t)*exp(-(t[i]-3*sita_t)*(t[i]-3*sita_t)/2/sita_t/sita_t);
    //  Wave[i]=exp(-(t[i]-3*sita_t)*(t[i]-3*sita_t)/2/sita_t/sita_t);
  dt[2*N_steps+1]=0.5*(dt[2*N_steps]*dt[2*N_steps]/dt[2*N_steps-1]+dt[2*N_steps]);
  t0=dt[0];
  dt[0]=0.5*(t0*t0/dt[1]+t0);
  for(i=1;i<2*N_steps+1;i++)
  {
    t1=dt[i];
    dt[i]=0.5*(t0+dt[i]);
    t0=t1;
  }

  FILE *Fp;
  if((Fp=fopen("testwaveform.dat","wb"))==NULL)
  {
    printf("fail to open file testwaveform\n");
    exit(EXIT_FAILURE);
  }

  fwrite(&N_steps,sizeof(int),1,Fp);
  fwrite(t,sizeof(double),2*N_steps+1,Fp);
  fwrite(dt,sizeof(double),2*N_steps+1,Fp);
  fwrite(Wave,sizeof(double),2*N_steps+1,Fp);
  fclose(Fp);

  free(t);
  return 0;


}
