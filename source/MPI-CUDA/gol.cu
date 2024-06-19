/*
 * Swarthmore College, CS 087 - Parallel & Distributed Computing
 *
 * Game of Life Implementation in CUDA
 * Dzineon Gyaltsen, Sean Cheng, Hoang Vu, Henry Lei
 * 11/23 - 12/23
 *
 */

#include "gol.h"
// #include <handle_cuda_error.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>

#include <sys/time.h>
#include <time.h>

static const int BLOCK_SIZE = 32; // pick so that it evenly divides N
static const int DEAD = 0;
static const int ALIVE = 1;

int *board;
int *temp_board;
int game_board_width_CPU;
int game_board_height_CPU;
int global_board_height_CPU;
int rank_temp;

/********** function prototypes  **************/
__global__ void play_gol(int *board_GPU, int *temp_board_GPU, int *numLiveCells, int *game_board_dim_width, int *game_board_dim_height);

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y, int *game_board_dim_width, int *game_board_dim_height);

CUDAKernel::CUDAKernel(int argc, char *argv[], int processed_args[], int rank, int size)
{
    int ret;
    rank_temp = rank;

    // Copy in process_args values into CUDAKernel equivalents
    game_board_width_CPU = processed_args[0];
    global_board_height_CPU = processed_args[1];
    // printf("global_board_height_CPU (ROUND 0) = %d\n", global_board_height_CPU);
    game_board_height_CPU = processed_args[4];
    // printf("game_board_height_CPU (ROUND 0) = %d\n", game_board_height_CPU);
    iterations = processed_args[2];
    oscillator = processed_args[3];

    // Allocate memory for CPU boards
    board = (int *)malloc(game_board_width_CPU * game_board_height_CPU * sizeof(int));
    temp_board = (int *)malloc(game_board_width_CPU * game_board_height_CPU * sizeof(int));

    // Initialize grid based on oscillator flag - if 0 every cell in grid is randomly alive or dead
    initialize_grid(rank, size);
    // printf("totalLiveCells (ROUND 0) = %d\n", totalLiveCells);

    // // cudaMalloc space on GPU for all relevant variables and boards
    // // cudaMalloc variables
    // ret = cudaMalloc((void **)&numLiveCells, sizeof(int));
    // if (ret != cudaSuccess)
    // {
    //     printf("malloc numLiveCells failed\n");
    //     exit(1);
    // }

    ret = cudaMalloc((void **)&game_board_dim_width, sizeof(int));
    if (ret != cudaSuccess)
    {
        printf("malloc game_board_dim_width failed\n");
        exit(1);
    }

    ret = cudaMalloc((void **)&game_board_dim_height, sizeof(int));
    if (ret != cudaSuccess)
    {
        printf("malloc game_board_dim_height failed\n");
        exit(1);
    }

    // cudaMalloc boards
    ret = cudaMalloc((void **)&board_GPU, sizeof(int) * game_board_width_CPU * game_board_height_CPU);
    if (ret != cudaSuccess)
    {
        printf("malloc board_GPU failed\n");
        exit(1);
    }

    ret = cudaMalloc((void **)&temp_board_GPU, sizeof(int) * game_board_width_CPU * game_board_height_CPU);
    if (ret != cudaSuccess)
    {
        printf("malloc temp_board_GPU failed\n");
        exit(1);
    }

    ret = cudaMalloc((void **)&top_row_GPU, sizeof(int) * game_board_width_CPU);
    if (ret != cudaSuccess)
    {
        printf("malloc top_row_GPU failed\n");
        exit(1);
    }

    ret = cudaMalloc((void **)&bottom_row_GPU, sizeof(int) * game_board_width_CPU);
    if (ret != cudaSuccess)
    {
        printf("malloc bottom_row_GPU failed\n");
        exit(1);
    }

    // // cudaMemcpy all relevant variables and boards from CPU to its GPU counterpart
    // // cudaMemcpy variables
    // ret = cudaMemcpy(numLiveCells, &totalLiveCells, sizeof(int),
    //                  cudaMemcpyHostToDevice);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy numLiveCells failed\n");
    //     cudaFree(numLiveCells);
    //     exit(1);
    // }

    ret = cudaMemcpy(game_board_dim_width, &game_board_width_CPU, sizeof(int),
                     cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("cudaMemcpy game_board_dim_width failed\n");
        cudaFree(game_board_dim_width);
        exit(1);
    }

    ret = cudaMemcpy(game_board_dim_height, &game_board_height_CPU, sizeof(int),
                     cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("cudaMemcpy game_board_dim_height failed\n");
        cudaFree(game_board_dim_height);
        exit(1);
    }

    // cudaMemcpy boards
    ret = cudaMemcpy(board_GPU, board, sizeof(int) * game_board_width_CPU * game_board_height_CPU,
                     cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("cudaMemcpy board_GPU failed\n");
        cudaFree(board_GPU);
        exit(1);
    }

    ret = cudaMemcpy(temp_board_GPU, temp_board, sizeof(int) * game_board_width_CPU * game_board_height_CPU,
                     cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("cudaMemcpy temp_board_GPU failed\n");
        cudaFree(temp_board_GPU);
        exit(1);
    }

    // if (rank_temp == 2) {

    //     print_board(1);

    // }
}

CUDAKernel::~CUDAKernel()
{
    int ret;

    // ret = cudaFree(numLiveCells);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaFree failed returned temp %d\n", ret);
    //     exit(1);
    // }
    // numLiveCells = nullptr;

    ret = cudaFree(game_board_dim_width);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    game_board_dim_width = nullptr;

    ret = cudaFree(game_board_dim_height);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    game_board_dim_height = nullptr;

    ret = cudaFree(board_GPU);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned board %d\n", ret);
        exit(1);
    }
    board_GPU = nullptr;

    ret = cudaFree(temp_board_GPU);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    temp_board_GPU = nullptr;

    ret = cudaFree(top_row_GPU);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    top_row_GPU = nullptr;

    ret = cudaFree(bottom_row_GPU);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    bottom_row_GPU = nullptr;
}

void CUDAKernel::update()
{

    // int ret;

    // define the block / thread block size
    dim3 blocks((game_board_width_CPU / BLOCK_SIZE), (game_board_height_CPU / BLOCK_SIZE), 1);
    dim3 threads_block(BLOCK_SIZE, BLOCK_SIZE, 1);

    /* call the play_gol kernel on world state */
    play_gol<<<blocks, threads_block>>>(board_GPU, temp_board_GPU, numLiveCells, game_board_dim_width, game_board_dim_height);

    // // swap boards by copying current GPU board to a CPU board, then back from CPU to the previous GPU board
    // ret = cudaMemcpy(temp_board, board_GPU, sizeof(int) * game_board_width_CPU * game_board_height_CPU,
    //                  cudaMemcpyDeviceToHost);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy temp_grid failed\n");
    //     cudaFree(board_GPU);
    //     exit(1);
    // }

    // ret = cudaMemcpy(temp_board_GPU, temp_board, sizeof(int) * game_board_width_CPU * game_board_height_CPU,
    //                  cudaMemcpyHostToDevice);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy temp_grid failed\n");
    //     cudaFree(temp_board_GPU);
    //     exit(1);
    // }

    // ret = cudaMemcpy(board, board_GPU, sizeof(int) * game_board_width_CPU * game_board_height_CPU,
    //                  cudaMemcpyDeviceToHost);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy temp_grid failed\n");
    //     cudaFree(temp_board_GPU);
    //     exit(1);
    // }

    // // update totalLiveCells per round by copying GPU numLiveCells value into CPU totalLiveCells
    // ret = cudaMemcpy(&totalLiveCells, numLiveCells, sizeof(int),
    //                  cudaMemcpyDeviceToHost);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy numLiveCells failed\n");
    //     cudaFree(numLiveCells);
    //     exit(1);
    // }

    for (int i = 0; i < game_board_width_CPU; i++)
    {

        top_row[i] = board[i];
    }

    for (int i = 0; i < game_board_width_CPU; i++)
    {

        bottom_row[i] = board[i + (game_board_width_CPU * (game_board_height_CPU - 1))];
    }

    // if (rank_temp == 2) {

    //     print_board(1);

    // }

    int *temp = board_GPU;
    board_GPU = temp_board_GPU;
    temp_board_GPU = temp;
}

__global__ void play_gol(int *board_GPU, int *temp_board_GPU, int *numLiveCells, int *game_board_dim_width, int *game_board_dim_height)
{
    int x, y, offset;
    // define offset
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    offset = x + y * *game_board_dim_width;

    // if valid cell in grid
    if ((x < *game_board_dim_width) && (y < *game_board_dim_height))
    {
        int num_neighbors = 0;
        num_neighbors = check_neighbors(board_GPU, temp_board_GPU, x, y, game_board_dim_width, game_board_dim_height);

        if (temp_board_GPU[offset] == ALIVE && (num_neighbors < 2 || num_neighbors > 3))
        {
            board_GPU[offset] = DEAD;
            // atomicSub(numLiveCells, 1);
        }

        else if (temp_board_GPU[offset] == DEAD && (num_neighbors == 3))
        {
            board_GPU[offset] = ALIVE;
            // atomicAdd(numLiveCells, 1);
        }
    }
}

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y, int *game_board_dim_width, int *game_board_dim_height)
{
    // define offset
    int offset = x + y * *game_board_dim_width;
    int num_neighbors = 0;

    // Check left
    if (x > 0)
    {
        int left = offset - 1;
        if (temp_board_GPU[left] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top left
    if (x > 0 && y > 0)
    {
        int topleft = offset - *game_board_dim_width - 1;
        if (temp_board_GPU[topleft] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top
    if (y > 0)
    {
        int top = offset - *game_board_dim_width;
        if (temp_board_GPU[top] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top right
    if (x < *game_board_dim_width - 1 && y > 0)
    {
        int topright = offset + 1 - *game_board_dim_width;
        if (temp_board_GPU[topright] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check right
    if (x < *game_board_dim_width - 1)
    {
        int right = offset + 1;
        if (temp_board_GPU[right] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom right
    if (y < *game_board_dim_height - 1 && x < *game_board_dim_width - 1)
    {
        int bottomright = offset + *game_board_dim_width + 1;
        if (temp_board_GPU[bottomright] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom
    if (y < *game_board_dim_height - 1)
    {
        int bottom = offset + *game_board_dim_width;
        if (temp_board_GPU[bottom] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom left
    if (y < *game_board_dim_height - 1 && x > 0)
    {
        int bottomleft = offset + *game_board_dim_width - 1;
        if (temp_board_GPU[bottomleft] == ALIVE)
        {
            num_neighbors++;
        }
    }

    return num_neighbors;
}

static void usage(void)
{

    fprintf(stderr,
            "./gol {-n board length -m board width -i iterations -o oscillator flag}\n"
            "-n          the board width\n"
            "-m          the board height\n"
            "-i          the number of iterations to run\n"
            "            must be > 1 to run\n"
            "-o          flag for oscillator pattern initialization\n"
            "-h          print out this message\n");
    exit(-1);
}

void CUDAKernel::process_args(int argc, char *argv[])
{
    int c = 0, n = -1, m = -1, i = -1, o = -1;

    while (1)
    {
        c = getopt(argc, argv, "n:m:i:o:");

        if (c == -1)
        {
            break;
        }

        switch (c)
        {
        case 'n':
            n = atoi(optarg);
            game_board_width_CPU = n;
            break;
        case 'm':
            m = atoi(optarg);
            game_board_height_CPU = m;
            break;
        case 'i':
            i = atoi(optarg);
            iterations = i;
            break;
        case 'o':
            o = atoi(optarg);
            oscillator = o;
            break;
        case 'h':
            usage();
            break;
        case ':':
            fprintf(stderr, "\n Error -%c missing arg\n", optopt);
            usage();
            break;
        case '?':
            fprintf(stderr, "\n Error unknown arg -%c\n", optopt);
            usage();
            break;
        }
    }
}

void CUDAKernel::print_board(int round)
{

    fprintf(stderr, "Round: %d\n", round);

    for (int i = 0; i < game_board_height_CPU; i++)
    { // going through grid
        for (int j = 0; j < game_board_width_CPU; j++)
        {
            //   fprintf(stderr, board[i][j]);
            if (board[j + i * game_board_width_CPU] == 1)
            {
                // if specified coordinate is alive, print as @
                fprintf(stderr, " @");
            }

            else
            { // if specified coordinate is dead, print as .
                fprintf(stderr, " .");
            }
        }
        fprintf(stderr, "\n");
    }

    usleep(200000 * 2);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
}

void CUDAKernel::initialize_grid(int rank, int size)
{

    if (oscillator == 1)
    {
        // initialize the rest of the grid
        for (int i = 0; i < game_board_height_CPU; i++)
        {
            for (int j = 0; j < game_board_width_CPU; j++)
            {
                board[i * game_board_width_CPU + j] = 0;
                temp_board[i * game_board_width_CPU + j] = 0;
            }
        }

        int center_row_global = global_board_height_CPU / 2;
        int first_row_global = rank * (global_board_height_CPU / size);

        // follow same logic as MPI to determine center of global grid vs local grid
        if (center_row_global >= first_row_global && center_row_global < first_row_global + game_board_height_CPU)
        {
            int center_row_local = center_row_global - first_row_global;
            int center_col = game_board_width_CPU / 2;
            board[(center_row_local + 3) * game_board_width_CPU + center_col - 1] = ALIVE;
            board[(center_row_local + 3) * game_board_width_CPU + center_col] = ALIVE;
            board[(center_row_local + 3) * game_board_width_CPU + center_col + 1] = ALIVE;
            temp_board[(center_row_local + 3) * game_board_width_CPU + center_col - 1] = ALIVE;
            temp_board[(center_row_local + 3) * game_board_width_CPU + center_col] = ALIVE;
            temp_board[(center_row_local + 3) * game_board_width_CPU + center_col + 1] = ALIVE;
            totalLiveCells = totalLiveCells + 3;
        }
    }
    else
    {
        srand(time(NULL) + rank);
        for (int i = 0; i < game_board_height_CPU; i++)
        {
            for (int j = 0; j < game_board_width_CPU; j++)
            {
                int k = rand() % 2;
                board[i * game_board_width_CPU + j] = k;
                temp_board[i * game_board_width_CPU + j] = k;
                if (k == 1)
                {
                    totalLiveCells++;
                }
            }
        }
    }
}
