/* Write2Files write the data to the files */
#include<stdio.h>
#include<stdlib.h>
#include"FTDT.h"
/*
typedef struct TimeSlot{
int N_slot;
double *Times;}TimeSlot;

typedef struct ReceiversPosition{
int N_Rec;
double *Rec_Pos_X;
double *Rec_Pos_Y;
double *Rec_Pos_Z;}Rec_Pos;

typedef struct DataFiles{
FILE *TimeSlotFile;
FILE *Record;
FILE *Inline;}DataFiles;
*/

int FilesPreparation(DataFiles *Files,TimeSlot *TSlot,Rec_Pos *Rec)
{
  if((Files->TimeSlotFile=fopen("TimeSlotData.dat","wb+"))==NULL)
  {
    printf("unable to create TimeSlotDataFile, press any key to exit!\n");
    getchar();
    exit(EXIT_FAILURE);
  }
  fwrite(&(TSlot->N_slot),sizeof(int),1,Files->TimeSlotFile);


  if((Files->Record=fopen("RecRecordedData.dat","wb+"))==NULL)
  {
    printf("unable to create RecRecordedData File, press any key to exit!\n");
    getchar();
    exit(EXIT_FAILURE);
  }
  fwrite(&(Rec->N_Rec),sizeof(int),1,Files->Record);
  fwrite(Rec->Rec_Pos_X,sizeof(double),Rec->N_Rec,Files->Record);
  fwrite(Rec->Rec_Pos_Y,sizeof(double),Rec->N_Rec,Files->Record);
  fwrite(Rec->Rec_Pos_Z,sizeof(double),Rec->N_Rec,Files->Record);

  if((Files->Inline=fopen("InlineArrayData.dat","wb+"))==NULL)
  {
    printf("unable to create InlineArray File, press any key to exit!\n");
    getchar();
    exit(EXIT_FAILURE);
  }

  return 0;

}

int Write2Files(DataFiles *Files,TimeSlot *TSlot,Rec_Pos *Rec,Efield *EF,Bfield *BF,ModelPara *MP,ModelGrid *MG)
{
  int i,j,k,L,M,N;

}
