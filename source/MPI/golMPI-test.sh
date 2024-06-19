#!/bin/bash

# default hostfile
# HOSTFILE=hostfile

# use this one for big runs
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
echo "using hostfile $HOSTFILE" >> ../MPI/outputs/MPIResults1.txt
sleep 2

for((n=2048; n <= 32768; n=n*2))
do
  for((p=2; p <= 32; p=p*2))
  do
    echo " "  >> ../MPI/outputs/MPIResults1.txt
    echo "#################" >> ../MPI/outputs/MPIResults1.txt
    echo "N: $n, I: 128, P: $p" >> ../MPI/outputs/MPIResults1.txt
    (time mpirun -np $p --hostfile ../MPI/$HOSTFILE ../MPI/golMPI -n $n -i 128 -o 1) &>> ../MPI/outputs/MPIResults1.txt
  done
done

################################################################################################
# # MPIResults2.txt (Testing maximum board size, Optimized Compiler)
# echo "using hostfile $HOSTFILE" >> ../MPI/outputs/MPIResults2.txt
# sleep 2

# for((n=32768*2; n <= 32768*16; n=n*2))
# do
#   echo " "  >> ../MPI/outputs/MPIResults2.txt
#   echo "#################" >> ../MPI/outputs/MPIResults2.txt
#   echo "N: $n, P: 32, I: 3" >> ../MPI/outputs/MPIResults2.txt
#   (time mpirun -np 32 --hostfile ../MPI/$HOSTFILE ../MPI/golMPI -n $n -i 3 -o 1) &>> ../MPI/outputs/MPIResults2.txt
# done

################################################################################################
# # MPIResults3.txt (Compare runtime to CUDA results, Optimized Compiler)
# echo "using hostfile $HOSTFILE" >> ../MPI/outputs/MPIResults3.txt
# sleep 2

# for((n=32000; n <= 40000; n=n+8000))
# do
#   for((i=1024; i <= 4096; i=i*2))
#   do
#     echo " "  >> ../MPI/outputs/MPIResults3.txt
#     echo "#################" >> ../MPI/outputs/MPIResults3.txt
#     echo "N: $n, I: $i, P: 32" >> ../MPI/outputs/MPIResults3.txt
#     (time mpirun -np 32 --hostfile ../MPI/$HOSTFILE ../MPI/golMPI -n $n -i $i -o 1) &>> ../MPI/outputs/MPIResults3.txt
#   done
# done