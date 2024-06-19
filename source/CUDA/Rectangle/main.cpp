/*
 * Swarthmore College, CS 087 - Parallel & Distributed Computing
 *
 * Game of Life Implementation in CUDA
 * Dzineon Gyaltsen, Sean Cheng, Hoang Vu, Henry Lei
 * 11/23 - 12/23
 *
 */

#include "gol.h"
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int totalLiveCells;

int main(int argc, char *argv[])
{
  int iters;

  totalLiveCells = 0;

  // Create cuda kernel
  CUDAKernel *kern = new CUDAKernel(argc, argv);

  // extract iterations information, then play for iters rounds
  iters = ((CUDAKernel *)kern)->getIters();

  for (int i = 0; i < iters; i++)
  {
    // kern->print_board(i);
    kern->update();
  }

  // printf("totalLiveCells (ROUND %d) = %d\n", iters, totalLiveCells);

  delete kern;

  return 0;
}
