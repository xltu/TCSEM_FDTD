/// a mathmatical function generate a array of log space
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
int logspace(double *Array, double begin, double end, int N)
{
  if(begin<=0)
  {
    printf("the begin of the logspace must be a positive number!\n");
    getchar();
    exit(EXIT_FAILURE);
  }

  double step_log;
  step_log=(log10(end)-log10(begin))/(N-1);
  for(int i=0;i<N;i++)
    Array[i]=pow10(log10(begin)+i*step_log);
  return 0;
}
int linspace(double *Array, double begin, double end, int N)
{
  double step;
  step=(end-begin)/(N-1);
  for(int i=0;i<N;i++)
    Array[i]=begin+i*step;
  return 0;
}


int searchInsert(double *A, int n, double target)
/*
Suppose we have a sorted array arr and a target value, we have to find the index when the target is found. 
If that is not present, then return the index where it 	would be if it were inserted in order.
So, if the input is like [1,3,4,6,6], and target = 5, then the output will be 3, 
as we can insert 5 at index 3, so the array will be [1,3,4,5,6,6]
*/
{
	if(n < 1) 
	{
     return 0;
  	}
  	int low = 0;
  	int high = n-1;
  	int mid;
  	int pos;
  	while(low <= high) 
  	{
		 mid = low + (high-low)/2;
		 if(A[mid] == target) 
		 {
			return mid;
		 }
		 else if(A[mid] > target) 
		 {
			high = mid - 1;
			pos = mid;
		 }
		 else 
		 {
			low = mid + 1;
			pos = mid + 1;
		 }
  	}
  	return pos;
}

