#!/bin/bash

################################################################################################
# NOTE: Use this command to fix [/bin/bash^M: bad interpreter: No such file or directory] Error
# sed -i -e 's/\r$//' runCUDA.sh
################################################################################################


################################################################################################
# CUDAResults1.txt (Debugging Compiler)
for((n=2048; n <= 32768; n=n*2))
do
  for ((i=128; i <= 1024; i=i*2))  
  do 
    echo " "  >> ../outputs/CUDAResults1.txt
    echo "#################" >> ../outputs/CUDAResults1.txt
    echo "N: $n, I: $i" >> ../outputs/CUDAResults1.txt
    (time ./gol -i $i -n $n -m $n -o 1) &>> ../outputs/CUDAResults1.txt 
  done

done

################################################################################################
# # CUDAResults2.txt (Optimized Compiler)
# for((n=2048; n <= 32768; n=n*2))
# do
#   for ((i=128; i <= 1024; i=i*2))  
#   do 
#     echo " "  >> ../outputs/CUDAResults2.txt
#     echo "#################" >> ../outputs/CUDAResults2.txt
#     echo "N: $n, I: $i" >> ../outputs/CUDAResults2.txt
#     (time ./gol -i $i -n $n -m $n -o 1) &>> ../outputs/CUDAResults2.txt 
#   done
# done

################################################################################################
# # CUDAResults3.txt (Testing Maximum Board Size, Optimized Compiler)
# for((n=33000; n <= 65000; n=n+8000))
# do
#   echo " "  >> ../outputs/CUDAResults3.txt
#   echo "#################" >> ../outputs/CUDAResults3.txt
#   echo "N: $n, I: 3" >> ../outputs/CUDAResults3.txt
#   (time ./gol -i 3 -n $n -m $n -o 1) &>> ../outputs/CUDAResults3.txt 
# done

# for((n=41000; n <= 49000; n=n+1000))
# do
#   echo " "  >> ../outputs/CUDAResults3.txt
#   echo "#################" >> ../outputs/CUDAResults3.txt
#   echo "N: $n, I: 3" >> ../outputs/CUDAResults3.txt
#   (time ./gol -i 3 -n $n -m $n -o 1) &>> ../outputs/CUDAResults3.txt 
# done

# for((n=45000; n <= 46000; n=n+200))
# do
#   echo " "  >> ../outputs/CUDAResults3.txt
#   echo "#################" >> ../outputs/CUDAResults3.txt
#   echo "N: $n, I: 3" >> ../outputs/CUDAResults3.txt
#   (time ./gol -i 3 -n $n -m $n -o 1) &>> ../outputs/CUDAResults3.txt 
# done

################################################################################################
# CUDAResults4.txt (Testing LARGE Runs, Optimized Compiler)
# OSCILLATOR FLAG
# for((n=32000; n <= 40000; n=n+8000))
# do
#   for ((i=1024; i <= 4096; i=i*2))  
#   do 
#     echo " "  >> ../outputs/CUDAResults4.txt
#     echo "#################" >> ../outputs/CUDAResults4.txt
#     echo "N: $n, I: $i" >> ../outputs/CUDAResults4.txt
#     (time ./gol -i $i -n $n -m $n -o 1) &>> ../outputs/CUDAResults4.txt 
#   done
# done

# echo "#####################################################################################" >> ../outputs/CUDAResults4.txt
# echo "#####################################################################################" >> ../outputs/CUDAResults4.txt

# # NO OSCILLATOR
# for((n=32000; n <= 40000; n=n+8000))
# do
#   for ((i=1024; i <= 4096; i=i*2))  
#   do 
#     echo " "  >> ../outputs/CUDAResults4.txt
#     echo "#################" >> ../outputs/CUDAResults4.txt
#     echo "N: $n, I: $i" >> ../outputs/CUDAResults4.txt
#     (time ./gol -i $i -n $n -m $n) &>> ../outputs/CUDAResults4.txt 
#   done
# done

################################################################################################
# # CUDAResultsDEMO.txt (Optimized Compiler)
# for((n=2048; n <= 16384; n=n*2))
# do
#   for ((i=128; i <= 1024; i=i*2))  
#   do 
#     echo " "  >> ../outputs/CUDAResultsDEMO.txt
#     echo "#################" >> ../outputs/CUDAResultsDEMO.txt
#     echo "N: $n, I: $i" >> ../outputs/CUDAResultsDEMO.txt
#     (time ./gol -i $i -n $n -m $n -o 1) &>> ../outputs/CUDAResultsDEMO.txt 
#   done

# done