#!/bin/bash

# default hostfile
# HOSTFILE=hostfile

# use this one for big runs (Results 2 + 3)
HOSTFILE=hostfilehuge

# optionly run with a hostfile command line arg
if [[ $# -gt 0 ]]; then
  HOSTFILE=$1
fi

################################################################################################
# NOTE: Use this command to fix [/bin/bash^M: bad interpreter: No such file or directory] Error
# sed -i -e 's/\r$//' golMPI-test.sh
################################################################################################

################################################################################################
# # MPIResults1.txt (Compare runtime to CUDA results, Optimized Compiler)
echo "using hostfile $HOSTFILE" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
sleep 2

for((n=32000; n <= 40000; n=n+8000))
do
  for((i=1024; i <= 2048; i=i*2))
  do
    echo " "  >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
    echo "#################" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
    echo "N: $n, I: $i, P: 32" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
    (time mpirun -np 32 --hostfile ../MPI-CUDA/$HOSTFILE ../MPI-CUDA/golMPI-CUDA -n $n -m $n -i $i -o 1) &>> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
  done
done

echo "####################################################################" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
echo "####################################################################" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt

# for((n=32768; n <= 40000; n=n+8000))
# do
# for((i=1024; i <= 2048; i=i*2))
# do
echo " "  >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
echo "#################" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
echo "N: 65536, I: 5, P: 32" >> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
(time mpirun -np 32 --hostfile ../MPI-CUDA/$HOSTFILE ../MPI-CUDA/golMPI-CUDA -n 65536 -m 65536 -i 5 -o 1) &>> ../MPI-CUDA/outputs/MPI-CUDAResults1.txt
# done
# done
