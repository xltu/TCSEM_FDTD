##################################################
# Program: TCSEM_FDTD
# 
# Include file for compilation 
#
# Environment: 
# Centos Linux at Parakepler at Oregon State Univ. 
#
#
# Created Nov.02.2020
# By Xiaoelei Tu
#
# Send bug reports, comments or suggestions
# to tuxl2009@hotmail.com
#
##################################################

# the path to gcc complier and its libraries
# MATLABPATH = /home/server/local/apps/matlab.2017b
FortBin = /usr/lib64
SysBin = /usr/lib/gcc/x86_64-redhat-linux/4.8.2
FDIR = /usr/bin
FFTBin = /home/parakepler/local/fftw-3.3.5
CUDABin = /opt/nvidia/hpc_sdk/Linux_x86_64/2020/cuda
cufftBin = /opt/nvidia/hpc_sdk/Linux_x86_64/20.9/math_libs/11.0/targets/x86_64-linux/

# FFTW pathes and bins
FFTINC = $(FFTBin)/api/ 
FFTLib = $(FFTBin)/.libs/
FFTNPLib = $(FFTBin)/threads/.libs/

# cufft pathes and bins
cufftINC = $(cufftBin)/include/
cufftLIB = $(cufftBin)/lib/

# 
# gcc compiler:
# 
CC = g++ -fopenmp

# Use this for faster runtime:
CFLAGS = -fPIC
COPTFLAGS = -m64 -Ofast -flto -march=native -funroll-loops
#CDEBUGFLAGS = -g
CDEBUGFLAGS =

LDFLAGS = -pthread $(COPTIMFLAGS) -Wl,-rpath-link,$(FFTLib) -L$(FFTLib) -lfftw3 -lm \
		  -L$(FFTNPLib) -lfftw3_omp 

INCLUDE = -I$(FFTINC) -I.

#
## NVCC compiler
#
NVCC = $(CUDABin)/bin/nvcc

# Use this for faster runtime:
NVFLAGS = --gpu-architecture=sm_37
NVOPTFLAGS = 
#NVDEBUGFLAGS = -g -G -lineinfo -DCUDA_DEVICE_ID=0
NVDEBUGFLAGS = -DCUDA_DEVICE_ID=0
NVCC_LIBS=

# CUDA Lib
CUDA_LIB_DIR= $(CUDABin)/lib64
CUDA_INC_DIR= -I$(CUDABin)/include -I$(cufftINC)
CUDA_LINK_LIBS= -lcudart
CUDA_LIBS= -Wl,-rpath=$(CUDA_LIB_DIR) -L$(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) \
				-Wl,-rpath=$(cufftLIB) -L$(cufftLIB) -lcufft -lcusparse


# You're done.  You shouldn't need to change anything below here.
# 
#----------------------------------------------------
# Build commands:
#----------------------------------------------------

TARGETS=  clean TCSEM_FDTD_GPU

OBJSOC= AuxiliaryMemory.o Bderivate.o  ElectricalFiled.o \
		check_fields_cuda.o DERF.o initialize_UpCont_gpu_resources.o\
		main.o ModelGridMapping.o helper_functions.o\
		SourceWave.o Spline.o SplineGridConvIR2R.o UpContinuation_gpu.o \
		EFInterp2Rx.o LoadRxTxPos.o initialize_cuda_device.o\
		logspace.o ModelingSetup.o prepare_free_mem.o initialize_fft.o \
		SplineGridConvIR2R_cpu.o
		
		
all:  $(TARGETS)
		
clean:	clean_msg
		rm -f *.o *~ core
		rm -f TCSEM_FDTD_GPU


TCSEM_FDTD_GPU: build_msg $(OBJSOC)
		$(CC) $(CFLAGS) $(COPTIMFLAGS) $(CDEBUGFLAGS) $(LDFLAGS) -o $@ $(OBJSOC) $(CUDA_LIBS)
			
#		
# Compile rules
#		

# General cpp compile:
%.o: %.cpp
	$(CC) $(INCLUDE) $(CUDA_INC_DIR) $(CFLAGS) $(COPTIMFLAGS) $(CDEBUGFLAGS) -c -o $@ $^	
	
# CUDA src file compile:
%.o: %.cu
	$(NVCC) $(INCLUDE) $(CUDA_INC_DIR) $(NVFLAGS) $(NVOPTFLAGS) $(NVDEBUGFLAGS) -c $< -o $@ $(NVCC_LIBS)		
	
#	
# Build Messages:
#	
clean_msg: 
	@printf "#\n# Cleaning files: \n#\n"
		
build_msg: 
	@printf "#\n# Building TCSEM_FDTD: \n#\n"
