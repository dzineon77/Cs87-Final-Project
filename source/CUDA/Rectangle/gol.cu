/*
 * Swarthmore College, CS 087 - Parallel & Distributed Computing
 *
 * Game of Life Implementation in CUDA
 * Dzineon Gyaltsen, Sean Cheng, Hoang Vu, Henry Lei
 * 11/23 - 12/23
 *
 */

#include "gol.h"
#include <handle_cuda_error.h>
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

/********** function prototypes  **************/
__global__ void init_rand(curandState *rand_state);

__global__ void play_gol(int *board_GPU, int *temp_board_GPU, int *numLiveCells, int *game_board_dim_width, int *game_board_dim_height);

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y, int *game_board_dim_width, int *game_board_dim_height);

CUDAKernel::CUDAKernel(int argc, char *argv[])
{
    int ret;

    // Obtain -n (rows), -m (columns), -i iterations, -o oscillator flag
    process_args(argc, argv);

    // Allocate memory for CPU boards
    board = (int *)malloc(game_board_width_CPU * game_board_height_CPU * sizeof(int));
    temp_board = (int *)malloc(game_board_width_CPU * game_board_height_CPU * sizeof(int));

    // Initialize grid based on oscillator flag - if 0 every cell in grid is randomly alive or dead
    initialize_grid();
    printf("totalLiveCells (ROUND 0) = %d\n", totalLiveCells);

    // cudaMalloc space on GPU for all relevant variables and boards
    // cudaMalloc variables
    ret = cudaMalloc((void **)&numLiveCells, sizeof(int));
    if (ret != cudaSuccess)
    {
        printf("malloc numLiveCells failed\n");
        exit(1);
    }

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

    // cudaMemcpy all relevant variables and boards from CPU to its GPU counterpart
    // cudaMemcpy variables
    ret = cudaMemcpy(numLiveCells, &totalLiveCells, sizeof(int),
                     cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("cudaMemcpy numLiveCells failed\n");
        cudaFree(numLiveCells);
        exit(1);
    }

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
}

CUDAKernel::~CUDAKernel()
{
    int ret;

    // Free all variables and boards that were cudaMalloc'd
    ret = cudaFree(numLiveCells);
    if (ret != cudaSuccess)
    {
        printf("cudaFree failed returned temp %d\n", ret);
        exit(1);
    }
    numLiveCells = nullptr;

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
}

void CUDAKernel::update()
{

    int ret;

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

    // // update totalLiveCells per round by copying GPU numLiveCells value into CPU totalLiveCells
    // ret = cudaMemcpy(&totalLiveCells, numLiveCells, sizeof(int),
    //                  cudaMemcpyDeviceToHost);
    // if (ret != cudaSuccess)
    // {
    //     printf("cudaMemcpy numLiveCells failed\n");
    //     cudaFree(numLiveCells);
    //     exit(1);
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
        // check each neighbor according to non-torus world
        num_neighbors = check_neighbors(board_GPU, temp_board_GPU, x, y, game_board_dim_width, game_board_dim_height);

        // if alive, check conditions to become dead
        if (temp_board_GPU[offset] == ALIVE && (num_neighbors < 2 || num_neighbors > 3))
        {
            board_GPU[offset] = DEAD;
            atomicSub(numLiveCells, 1);
        }

        // if dead, check conditions to become alive
        else if (temp_board_GPU[offset] == DEAD && (num_neighbors == 3))
        {
            board_GPU[offset] = ALIVE;
            atomicAdd(numLiveCells, 1);
        }
    }
}

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y, int *game_board_dim_width, int *game_board_dim_height)
{
    // define offset
    int offset = x + y * *game_board_dim_width;
    int num_neighbors = 0;

    // Check left neighbor
    if (x > 0)
    {
        int left = offset - 1;
        if (temp_board_GPU[left] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top left neighbor
    if (x > 0 && y > 0)
    {
        int topleft = offset - *game_board_dim_width - 1;
        if (temp_board_GPU[topleft] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top neighbor
    if (y > 0)
    {
        int top = offset - *game_board_dim_width;
        if (temp_board_GPU[top] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check top right neighbor
    if (x < *game_board_dim_width - 1 && y > 0)
    {
        int topright = offset + 1 - *game_board_dim_width;
        if (temp_board_GPU[topright] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check right neighbor
    if (x < *game_board_dim_width - 1)
    {
        int right = offset + 1;
        if (temp_board_GPU[right] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom right neighbor
    if (y < *game_board_dim_height - 1 && x < *game_board_dim_width - 1)
    {
        int bottomright = offset + *game_board_dim_width + 1;
        if (temp_board_GPU[bottomright] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom neighbor
    if (y < *game_board_dim_height - 1)
    {
        int bottom = offset + *game_board_dim_width;
        if (temp_board_GPU[bottom] == ALIVE)
        {
            num_neighbors++;
        }
    }

    // Check bottom left neighbor
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
            "./gol {-n board length/width -i iterations | -f filename}\n"
            "-n          the board width\n"
            "-m          the board height\n"
            "-i          the number of iterations to run\n"
            "            must be > 1 to run\n"
            "-o          set oscillator flag\n");
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
            // n is board width parameter
            game_board_width_CPU = n;
            break;
        case 'm':
            m = atoi(optarg);
            // m is board height parameter
            game_board_height_CPU = m;
            break;
        case 'i':
            i = atoi(optarg);
            // i is iterations parameter
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
            if (temp_board[i * game_board_width_CPU + j] == 1)
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

    usleep(100000 * 2);
}

void CUDAKernel::initialize_grid()
{

    if (oscillator == 1)
    {
        // first initialize the whole grid to dead,
        for (int i = 0; i < game_board_width_CPU; i++)
        {
            for (int j = 0; j < game_board_height_CPU; j++)
            {
                board[i + j * game_board_width_CPU] = 0;
                temp_board[i + j * game_board_width_CPU] = 0;
            }
        }

        // define center coordinates of board,
        center[0] = game_board_width_CPU / 2;
        center[1] = game_board_height_CPU / 2;

        // then set oscillator pattern to alive at the center of the board
        board[center[0] + center[1] * game_board_width_CPU] = ALIVE;
        board[center[0] + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        board[center[0] + (center[1] - 1) * game_board_width_CPU] = ALIVE;
        temp_board[center[0] + center[1] * game_board_width_CPU] = ALIVE;
        temp_board[center[0] + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        temp_board[center[0] + (center[1] - 1) * game_board_width_CPU] = ALIVE;

        totalLiveCells = totalLiveCells + 3;

        // // Non-Torus demo
        // board[center[0] + center[1] * game_board_width_CPU] = ALIVE;
        // board[center[0] + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        // board[center[0] + (center[1] - 1) * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + center[1] * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + (center[1] - 1) * game_board_width_CPU] = ALIVE;

        // totalLiveCells = totalLiveCells + 3;

        // board[center[0] + (game_board_width_CPU / 2) - 1 + center[1] * game_board_width_CPU] = ALIVE;
        // board[center[0] + (game_board_width_CPU / 2) - 1 + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        // board[center[0] + (game_board_width_CPU / 2) - 1 + (center[1] - 1) * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + (game_board_width_CPU / 2) - 1 + center[1] * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + (game_board_width_CPU / 2) - 1 + (center[1] + 1) * game_board_width_CPU] = ALIVE;
        // temp_board[center[0] + (game_board_width_CPU / 2) - 1 + (center[1] - 1) * game_board_width_CPU] = ALIVE;

        // totalLiveCells = totalLiveCells + 3;

        // board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 3) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 4) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 6) + (center[1] - (game_board_height_CPU / 2) + 4) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 7) + (center[1] - (game_board_height_CPU / 2) + 3) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 2) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 3) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 4) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 6) + (center[1] - (game_board_height_CPU / 2) + 4) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 7) + (center[1] - (game_board_height_CPU / 2) + 3) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 5) + (center[1] - (game_board_height_CPU / 2) + 2) * game_board_width_CPU] = ALIVE;

        // totalLiveCells = totalLiveCells + 5;

        // board[(center[0] + (game_board_width_CPU / 2) - 2) + (center[1] + (game_board_height_CPU / 2) - 1) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 2) + (center[1] + (game_board_height_CPU / 2) - 2) * game_board_width_CPU] = ALIVE;
        // board[(center[0] + (game_board_width_CPU / 2) - 1) + (center[1] + (game_board_height_CPU / 2) - 2) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 2) + (center[1] + (game_board_height_CPU / 2) - 1) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 2) + (center[1] + (game_board_height_CPU / 2) - 2) * game_board_width_CPU] = ALIVE;
        // temp_board[(center[0] + (game_board_width_CPU / 2) - 1) + (center[1] + (game_board_height_CPU / 2) - 2) * game_board_width_CPU] = ALIVE;

        // totalLiveCells = totalLiveCells + 3;
    }
    else
    {
        srand(time(NULL));
        int k;

        // otherwise randomly initialize the rest of the grid to alive or dead
        for (int i = 0; i < game_board_width_CPU; i++)
        {
            for (int j = 0; j < game_board_height_CPU; j++)
            {
                k = rand() % 2;
                if (k == 1)
                {
                    totalLiveCells = totalLiveCells + 1;
                }
                board[i + j * game_board_width_CPU] = k;
                temp_board[i + j * game_board_width_CPU] = k;
            }
        }
    }
}
