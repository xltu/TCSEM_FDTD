#ifndef FTDT_H_INCLUDED
#define FTDT_H_INCLUDED

#include <cuda_runtime.h>
#include <cuda.h>

/* ----------------------------------------------------------------------------------------------- */
// for debugging and benchmarking
/* ----------------------------------------------------------------------------------------------- */

#define DEBUG 1
#if DEBUG == 1
#define TRACE(x) printf("%s\n",x);
#else
#define TRACE(x) // printf("%s\n",x);
#endif

#define MAXDEBUG 1
#if MAXDEBUG == 1
#define LOG(x) printf("%s\n",x)
#define PRINT5(var,offset) for(;print_count<5;print_count++) printf("var(%d)=%2.20f\n",print_count,var[offset+print_count]);
#define PRINT10(var) if (print_count<10) { printf("var=%1.20e\n",var); print_count++; }
#define PRINT10i(var) if (print_count<10) { printf("var=%d\n",var); print_count++; }
#else
#define LOG(x) // printf("%s\n",x);
#define PRINT5(var,offset) // for(i=0;i<10;i++) printf("var(%d)=%f\n",i,var[offset+i]);
#endif


// maximum function
#define max(x,y)    ( ((x) < (y)) ? (y) : (x) )
// minimum function
#define min(a,b)    ( ((a) > (b)) ? (b) : (a) )

//define USE_SINGLE_PRECISION

#define PI 3.141592653589793238
// minimum loop size to initiate openMP
#define MINLOOP 124	
// maximum length of file name
#define LFILE 200		

// type of "working" variables
#ifdef USE_SINGLE_PRECISION
typedef float realw;  
#else
typedef double realw;
#endif

// magnetic permeability in free space
extern double miu;
extern int MAXTHREAD;

/* ----------------------------------------------------------------------------------------------- */
// field pointer wrapper structure
/* ----------------------------------------------------------------------------------------------- */

typedef struct DeviceArrays {
  cudaPitchedPtr Ex;
  cudaPitchedPtr Ey;
  cudaPitchedPtr Ez;
  cudaPitchedPtr Bx;
  cudaPitchedPtr By;
  cudaPitchedPtr Bz;
  cudaPitchedPtr Con;
  realw *BxAir;					//XXX should be removed in a future version
  realw *ByAir;					//XXX should be removed in a future version
} DArrays;

#define MP_VER_1

typedef struct ModelParameter{
#ifdef MP_VER_1
  int AnLx_old;			// mesh size variables in 1.0 version code
  int AnRx_old;
  int Any_old;
  int Anz;
#endif  
  int ALx1;		// mesh size variables in 2.0 version code
  int ALx2;
  int ARx1;
  int ARx2;
  int Ay1;
  int Ay2;
  int Az1;
  int Az2;
  int L;
  int M;
  int N;
  
  int Nx;	// shared variables in 1.0, and after
  int Ny;
  int Nz;
  int AnLx_New;
  int AnRx_New;
  int Any_New;
  int LNx2;
  int LMy2;
  realw dxmin;
  realw dymin;
  realw dzmin;
  realw arg1;		// variables in 2.0 version code
  realw arg2;
}ModelPara;

typedef struct ModelGrid{
  realw *dx;
  realw *dy;
  realw *dz;
  realw *Z;
  realw *X_Bzold;
  realw *Y_Bzold;
  realw *X_BzNew;
  realw *Y_BzNew;
}ModelGrid;

#define FFTW
#ifdef FFTW

#include<fftw3.h>
typedef struct UpContPara{
fftw_plan plan1;
fftw_plan plan2;
fftw_plan plan3;
double **Bz0;
double **BzAir;
double **BxAir;
double **ByAir;
fftw_complex **FBzC;
fftw_complex **FBxC;
fftw_complex **FByC;
double **FBxR;
double **FByR;
}UpContP;

#endif

typedef struct GridConversion{
  int *xBzNew2old;
  int *yBzNew2old;
  int *xBzold2New;
  int *yBzold2New;
  int *xoldBx2NewBz;
  int *yoldBy2NewBz;
}GridConv;


// structure for recording time
typedef struct RecTime{
int Nt;
int IsLog;
int NComp;		//total number of components
int nefcmp;		//number of E component
int isComp[6];
realw T0;
realw Tn;
realw *Times;
}RxTime;

// receiver positions
typedef struct ReceiversPos{
realw *x;
realw *y;
realw *z;
int *ix;
int *iy;
int *iz;
}RxPos;

// bipole transmitter positions
typedef struct TransmitterPos{
int ort;
int nr;
realw BipoLen;
realw x;
realw y;
realw z;
int ixa;
int iya;
int iza;
int ixb;
int iyb;
int izb;
int *RxNum;	// the #Rx associated with the current transmitter positions
}TxPos;


double DERF(double X);
double DERFC(double X);
double DERFCX(double X);

int logspace(double *Array, double begin, double end, int N);
int linspace(double *Array, double begin, double end, int N);
int searchInsert(double *A, int n, double target);
double Spline3D4Ps(double *X, double *Y, double *Z, double (*data)[4][4], double x, double y, double z);

#ifdef FFTW
fftw_complex **Create2DfftwArray(int ,int );
void Free2DfftwArray(fftw_complex **);
#endif

void Free2DArray(double **);
double **Create2DArray(int,int);
double ***Create_3D_Array(int,int,int);
int Free_3D_Array(double ***,int);

int ModelingSetup(double ****,ModelPara *,ModelGrid *,double *,double *,double *,int *,double *);
//void Model(int,int,int,int,int,int,int,int,int,int,int);

//int LoadModel(double ***,ModelPara *,ModelGrid *);

int Waveform(int N_steps,double sita_t,double *dt,double *Wave);

//int LoadInitialValue(double ***,Efield *,Bfield *,ModelGrid *,ModelPara *,GridConv *,
//                     UpContP *,double,double);


int Bderivate( DArrays *, ModelPara , ModelPara *, ModelGrid *, realw );

//int CurMagneticField(double ***,double ***,double ***,double ***,double ***,double ***,
//                     double **,double **,ModelPara *,ModelGrid *,double,double);

int ElectricalField( DArrays *, ModelPara *, ModelPara *, ModelGrid *,	realw,	realw,	realw,	TxPos, TxPos *);

int ModelGridMapping(ModelPara *,ModelGrid *,GridConv *);

int SplineGridConvIR2R(ModelPara *,ModelGrid *,GridConv *,double **,double **);

int BilinearGridConvR2IR(int,ModelPara *,ModelGrid *,GridConv *,double **,double **);

int UpContinuation(DArrays *, UpContP *,ModelPara *,ModelGrid *,GridConv *);



int LoadRxTxPos(ModelPara, ModelGrid, RxTime *, RxPos *, TxPos **, int *, int *);

/*
int EFInterp2Rx(int *ne_efs, int *nb_efs, RxTime tRx, TxPos xyzTx, RxPos *xyzRx, ModelGrid MG, Efield *EF,Bfield *BF,\
				 double t0, double dt, double (*Tpof)[3], double ***, double ***);
				 
int EFXInterp(ModelGrid MG, Efield *EF, RxTime tRx, TxPos xyzTx, RxPos *xyzRx, double **ef0);
int BFXInterp(ModelGrid MG, Bfield *BF, RxTime tRx, TxPos xyzTx, RxPos *xyzRx, double **ef0);
*/

int EFInterp2Rx(int *ne_efs, int *nb_efs, RxTime tRx, TxPos xyzTx, RxPos *d_xyzRx, int *RxLst,
								ModelGrid *d_MG, ModelPara *d_mp, DArrays *DPtrs, realw t0, realw dt, 
								realw (*Tpof)[3], realw *efs, realw *ef0);
																
				 
int FreeRxTxArray(RxTime *, RxPos *, TxPos **, int , int );

void prepare_device_arrays(DArrays* DPtrs, ModelPara MP, realw ***HostCond, 
													 RxPos *d_xyzRx, RxPos xyzRx, int NRx, int **RxLst,
													 ModelGrid *d_MG, ModelGrid MG);
													 
void set_zeros_EM_arrays(DArrays *, ModelPara *);

void Device_cleanup(DArrays *, RxPos *, int **, ModelGrid *) ;	

#ifdef FFTW
void CP2Host_bz0_UPP(cudaPitchedPtr Bz, realw *bz0, int M, int L);
void CP2Device_bxby_UPP(DArrays* DPtrs, realw *bx0, realw *by0, int M, int L);
#endif												 

#endif // FTDT_H_INCLUDED
