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

static const int BLOCK_SIZE = 8;     // pick so that it evenly divides N
static const int DEAD = 0;
static const int ALIVE = 1;

static int board[N][N];
static int temp_board[N][N];

/********** function prototypes  **************/
__global__ void init_rand(curandState *rand_state);

__global__ void play_gol(int *board_GPU, int *temp_board_GPU, int *numLiveCells);

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y);

CUDAKernel::CUDAKernel(int argc, char *argv[]) {
  int ret;
  char *fileName = NULL;

  process_args(argc, argv, &fileName);
  initialize_grid();
  printf("totalLiveCells (ROUND 0) = %d\n", totalLiveCells);
  
  ret = cudaMalloc((void **) &numLiveCells, sizeof(int)); 
  if (ret != cudaSuccess) {
    printf("malloc board failed\n");
    exit(1);
    }

  // Malloc board space on GPU
  ret = cudaMalloc((void **) &board_GPU, sizeof(int) * N * N);
  if (ret != cudaSuccess) {
      printf("malloc board failed\n");
      exit(1);
  }

  // Malloc board space on GPU
  ret = cudaMalloc((void **) &temp_board_GPU, sizeof(int) * N * N);
  if (ret != cudaSuccess) {
      printf("malloc temp_board failed\n");
      exit(1);
  }

  ret = cudaMemcpy(numLiveCells, &totalLiveCells, sizeof(int),
                   cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy temp_grid failed\n");
      cudaFree(board_GPU);
      exit(1);
  }

  // copy the initial host data (host board) to the GPU
  ret = cudaMemcpy(board_GPU, board, sizeof(int) * N * N,
                   cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy temp_grid failed\n");
      cudaFree(board_GPU);
      exit(1);
  }

  // copy the initial host data (host board) to the GPU
  ret = cudaMemcpy(temp_board_GPU, temp_board, sizeof(int) * N * N,
                   cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy status_grid failed\n");
      cudaFree(temp_board_GPU);
      exit(1);
  }

}

CUDAKernel::~CUDAKernel() {
  int ret;
  
  ret = cudaFree(numLiveCells);
  if (ret != cudaSuccess) {
      printf("cudaFree failed returned temp %d\n", ret);
      exit(1);
  }
  numLiveCells = nullptr;

  ret = cudaFree(temp_board_GPU);
  if (ret != cudaSuccess) {
      printf("cudaFree failed returned temp %d\n", ret);
      exit(1);
  }
  temp_board_GPU = nullptr;

  ret = cudaFree(board_GPU);
  if (ret != cudaSuccess) {
      printf("cudaFree failed returned board %d\n", ret);
      exit(1);
  }
  board_GPU = nullptr;

}

void CUDAKernel::update() {
  
  int ret;

  // define the block / thread block size
  dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE, 1);
  dim3 threads_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  /* call the play_gol kernel on world state */
  play_gol<<< blocks, threads_block >>>(board_GPU, temp_board_GPU, numLiveCells);
  
  // swap boards
  ret = cudaMemcpy(temp_board, board_GPU, sizeof(int) * N * N,
                   cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy temp_grid failed\n");
      cudaFree(board_GPU);
      exit(1);
  }

  ret = cudaMemcpy(temp_board_GPU, temp_board, sizeof(int) * N * N,
                   cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy temp_grid failed\n");
      cudaFree(temp_board_GPU);
      exit(1);
  }
  
  ret = cudaMemcpy(&totalLiveCells, numLiveCells, sizeof(int),
                   cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess) {
      printf("cudaMemcpy numLiveCells failed\n");
      cudaFree(numLiveCells);
      exit(1);
  }

}

__global__ void play_gol(int *board_GPU, int *temp_board_GPU, int *numLiveCells) {
  int x, y, offset;

  x = blockIdx.x * blockDim.x + threadIdx.x;
  y = blockIdx.y * blockDim.y + threadIdx.y;
  offset = x + y * N;

  if ((x < N) && (y < N)) {
    int num_neighbors = 0;
    num_neighbors = check_neighbors(board_GPU, temp_board_GPU, x, y);

    if (temp_board_GPU[offset] == ALIVE && (num_neighbors < 2 || num_neighbors > 3)) {
      board_GPU[offset] = DEAD;
      atomicSub(numLiveCells, 1);
    }

    else if (temp_board_GPU[offset] == DEAD && (num_neighbors == 3)) {
      board_GPU[offset] = ALIVE;
      atomicAdd(numLiveCells, 1);
    }

  }
}

__device__ int check_neighbors(int *board_GPU, int *temp_board_GPU, int x, int y) {

  int offset = x + y * N;
  int num_neighbors = 0;

  // Check left
  if (x > 0) {
      int left = offset - N;
      if (temp_board_GPU[left] == ALIVE) {
          num_neighbors++;
      }
  }
  
  // Check top left
  if (x > 0 && y > 0) {
      int topleft = offset - N - 1;
      if (temp_board_GPU[topleft] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check top
  if (y > 0) {
      int top = offset - 1;
      if (temp_board_GPU[top] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check top right
  if (x < N - 1 && y > 0) {
      int topright = offset + N - 1;
      if (temp_board_GPU[topright] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check right
  if (x < N - 1) {
      int right = offset + N;
      if (temp_board_GPU[right] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check bottom right
  if (y < N - 1 && x < N - 1) {
      int bottomright = offset + N + 1;
      if (temp_board_GPU[bottomright] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check bottom
  if (y < N - 1) {
      int bottom = offset + 1;
      if (temp_board_GPU[bottom] == ALIVE) {
          num_neighbors++;
      }
  }

  // Check bottom left
  if (y < N - 1 && x > 0) {
      int bottomleft = offset - N + 1;
      if (temp_board_GPU[bottomleft] == ALIVE) {
          num_neighbors++;
      }
  }

  return num_neighbors;

}

static void usage(void) {

    fprintf(stderr,
            "./gol {-n board length/width -i iterations | -f filename}\n"
            "-n          the board length and width, square with dimensions n x n\n"
            " -i         the number of iterations to run\n"
            "            must be > 1 to run\n"
            "-f          filename read in configuration info from a file\n"
            "-h          print out this message\n"
    );
    exit(-1);
}

void CUDAKernel::process_args(int argc, char *argv[], char **fileName) {
  int c = 0, n = -1, i = -1, o = -1;

  while (1) {
      c = getopt(argc, argv, "n:i:o:f:");

      if (c == -1) {
          break;
      }

      switch (c) {
          case 'n':
              n = atoi(optarg);
              game_board_dim = n;
              break;
          case 'i':
              i = atoi(optarg);
              iterations = i;
              break;
          case 'o':
              o = atoi(optarg);
              oscillator = o;
              break;
          case 'f':
              *fileName = optarg;
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

  if (*fileName == NULL) {
      if (n == -1) {
          n = N;
      }
      if (i == -1) {
          i = NUMITERS;
      }
      if (o == -1) {
          o = 1;
      }
  }

  if (*fileName != NULL && (n != -1 || i != -1)) {
      fprintf(stderr, "Error: cannot be run with both -f and -i -n\n");
      usage();
  }

    if (*fileName != NULL) {
        FILE *infile;
        infile = fopen(*fileName, "r");

        if (infile == NULL) {
            printf("Error: file open %s\n", *fileName);
            exit(1);
        }
        
        // read from file
        int ret = fscanf(infile, "%d%d%d", &game_board_dim, &iterations, &totalLiveCells);
        // printf("game_board_dim = %d, iterations = %d, totalLiveCells = %d\n", game_board_dim, iterations, totalLiveCells);

        for (int i = 0; i < totalLiveCells; i++) {
            int posx = 0;
            int posy = 0;
            int ret = fscanf(infile, "%d%d", &posx, &posy);
            board[posx][posy] = ALIVE;
            temp_board[posx][posy] = ALIVE;
        }

        fclose(infile);
    }

}

void CUDAKernel::print_board(int round) {

  fprintf(stderr, "Round: %d\n", round);
  
  for(int i = 0; i < N; i++){ //going through grid
    for(int j = 0; j < N; j++){
    //   fprintf(stderr, board[i][j]);
       if(temp_board[i][j] == 1){
         //if specified coordinate is alive, print as @
          fprintf(stderr, " @");}
       
       else {//if specified coordinate is dead, print as .
          fprintf(stderr, " .");}
  
    }
      fprintf(stderr, "\n");
    }
  
    usleep(200000 * 2);
}

void CUDAKernel::initialize_grid() {

  if (oscillator == 1) {
    // initialize center
    center[0] = N / 2;
    center[1] = N / 2;

    // oscillator
    board[center[0]][center[1]] = ALIVE;
    board[center[0]][center[1]+1] = ALIVE;
    board[center[0]][center[1]-1] = ALIVE;
    temp_board[center[0]][center[1]] = ALIVE;
    temp_board[center[0]][center[1]+1] = ALIVE;
    temp_board[center[0]][center[1]-1] = ALIVE;

    totalLiveCells = totalLiveCells + 3;

    // // square block
    // board[center[0]][center[1]] = ALIVE;
    // board[center[0]+1][center[1]] = ALIVE;
    // board[center[0]+1][center[1]+1] = ALIVE;
    // board[center[0]][center[1]+1] = ALIVE;

    // temp_board[center[0]][center[1]] = ALIVE;
    // temp_board[center[0]+1][center[1]] = ALIVE;
    // temp_board[center[0]+1][center[1]+1] = ALIVE;
    // temp_board[center[0]][center[1]+1] = ALIVE;

    // totalLiveCells = totalLiveCells + 4;
  }
  else { 
    srand(time(NULL));
    int k;

    // initialize the rest of the grid
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          k = rand() % 2;
          if (k == 1) {
            totalLiveCells = totalLiveCells + 1;
          }
          board[i][j] = k;
          temp_board[i][j] = k;
        }
      }
    }

  }
