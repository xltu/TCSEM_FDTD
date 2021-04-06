#!/bin/sh                                                                       
#                                                                               
# Run set up a TCSEM_FDTD modeling                                                        
#                                                                               
## From the command lscpu                                                       
## parakepler has 56 CPUs: 14 cores per socket, 2 sockets, 2 threads per core.  
## CPU max clock is 3.6 GHz                                                     
## check memory use with command free                                           
## first the basics to get it started. nthreads should be 2*nfreq+1             
#set mpi = '/opt/pgi/linux86-64/current/mpi/openmpi-3.1.3/bin/mpirun'           
#MPI='mpiexec'                                                                   
#execute='Mod3DMT'                                                               
#nthreads='13'                                                                   
## and the details for this particular run                                      
                                                
#if (-f $out) then                                                              
#    echo Removing previous inv.out                                             
#    rm $out                                                                    
#endif     
export OMP_NUM_THREADS=12    
export OMP_SCHEDULE="guided,32"       
#export OMP_DYNAMIC=TRUE
export OMP_NESTED=TRUE  
                                                               
echo TCSEM_FDTD  > FDTD.log &
nohup TCSEM_FDTD > FDTD.log &
