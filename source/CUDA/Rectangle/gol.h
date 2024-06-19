/*
 * Swarthmore College, CS 087 - Parallel & Distributed Computing
 *
 * Game of Life Implementation in CUDA
 * Dzineon Gyaltsen, Sean Cheng, Hoang Vu, Henry Lei 
 * 11/23 - 12/23
 *
 */

#pragma once

#include <curand_kernel.h>
#include <curand.h>
#include <cuda.h>

static const int N = 64;          // dimension of square world
static const int NUMITERS = 10; // default values

extern int totalLiveCells;

class CUDAKernel
{

public:
    CUDAKernel(int argc, char *argv[]);
    ~CUDAKernel();
    void update();
    void print_board(int round);
    int getIters() { return iterations; }

private:
    // Variables
    int *board_GPU;
    int *temp_board_GPU;
    int *game_board_dim_width;
    int *game_board_dim_height;
    int iterations;
    int oscillator = 0;
    int center[2];
    int *numLiveCells;

    // Functions
    void process_args(int argc, char *argv[]);
    void initialize_grid();
};
