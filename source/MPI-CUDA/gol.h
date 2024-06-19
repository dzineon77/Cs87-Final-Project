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

static const int N = 60;          // dimension of square world
static const int NUMITERS = 5; // default values

extern int totalLiveCells;
extern int *top_row;
extern int *bottom_row;

class CUDAKernel
{

public:
    CUDAKernel(int argc, char *argv[], int processed_args[], int rank, int size);
    ~CUDAKernel();
    void update();
    // int *getGrid();
    // void copyTopRowfromGPU(int *board_GPU, int *top_row, int width);
    // void copyBottomRowfromGPU(int *board_GPU, int *bottom_row, int width);

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
    int *top_row_GPU;
    int *bottom_row_GPU;

    // Functions
    void process_args(int argc, char *argv[]);
    void initialize_grid(int rank, int size);
};
