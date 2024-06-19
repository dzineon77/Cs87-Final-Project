#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/time.h>
#include <time.h>
#include "gol.h"

// #define N 60 // Grid dimensions
// #define ITERS 5
#define ALIVE 1
#define DEAD 0

// Function prototypes
void initialize_my_grid(int *grid, int rows, int cols, int rank, int size, int oscillatorFlag);
int play_gol(int *my_grid, int *new_grid, int my_rows, int rank, int size, int *top_row, int *bottom_row);
int count_alive_neighbors(int *grid, int x, int y, int rows, int cols, int rank, int size, int *top_row, int *bottom_row);
int count_live_cells(int *grid, int rows, int cols);
void process_args(int argc, char *argv[]);
void print_board(int iter, int my_rows, int *my_grid);

int width = -1, height = -1, i = -1, o = -1;
int totalLiveCells = 0;
int processed_args[5];

int *top_row;
int *bottom_row;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Obtain -n (rows), -m (columns), -i iterations, and -o oscillator flag,
    // and place in buffer to send to cuda kernel
    process_args(argc, argv);
    processed_args[0] = width;
    processed_args[1] = height;
    processed_args[2] = i;
    processed_args[3] = o;

    // Calculate rows per process and allocate memory for the local grid
    int rows_per_process = height / size;
    int extra_rows = height % size;
    int my_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    processed_args[4] = my_rows;

    // int *my_grid = (int *)malloc(my_rows * N * sizeof(int));
    // int *new_grid = (int *)malloc(my_rows * N * sizeof(int));
    top_row = (int *)malloc(width * sizeof(int));
    bottom_row = (int *)malloc(width * sizeof(int));

    CUDAKernel *kern = new CUDAKernel(argc, argv, processed_args, rank, size);

    // int localLiveCells = totalLiveCells;
    // printf("localLiveCells = %d\n", localLiveCells);

    int globalLiveCells;
    MPI_Reduce(&totalLiveCells, &globalLiveCells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (rank == 0)
    // {
    //     printf("\nglobalLiveCells BEFORE = %d\n", globalLiveCells);
    // }

    // Main loop for Game of Life iterations
    for (int iter = 0; iter < i; iter++)
    {

        // Communicate boundary rows
        MPI_Status status;
        if (rank != 0)
        {
            MPI_Sendrecv(top_row, width, MPI_INT, rank - 1, 0,
                         bottom_row, width, MPI_INT, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
        }
        if (rank != size - 1)
        {
            MPI_Sendrecv(bottom_row, width, MPI_INT, rank + 1, 1,
                         top_row, width, MPI_INT, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        }

        kern->update();
        MPI_Reduce(&totalLiveCells, &globalLiveCells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // if (rank == 1)
        // {
        //     // print_board(iter, my_rows, my_grid);
        //     for (int i = 0; i < width; i++) {
        //         printf("%d", top_row[i]);
        //     }
        // }

        // localLiveCells = play_gol(my_grid, new_grid, my_rows, rank, size, top_row, bottom_row);
        // MPI_Reduce(&localLiveCells, &globalLiveCells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // // Swap grids for the next iteration
        // int *temp = my_grid;
        // my_grid = new_grid;
        // new_grid = temp;
    }

    // Cleanup and finalize MPI
    // free(my_grid);
    // free(new_grid);
    free(top_row);
    free(bottom_row);
    delete kern;
    MPI_Finalize();
    // if (rank == 0)
    // {
    //     printf("\nglobalLiveCells AFTER = %d\n", globalLiveCells);
    // }
    return 0;
}

int play_gol(int *my_grid, int *new_grid, int my_rows, int rank, int size, int *top_row, int *bottom_row)
{
    int localLiveCells = 0;

    for (int i = 0; i < my_rows; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int alive_neighbors = count_alive_neighbors(my_grid, i, j, my_rows, N, rank, size, top_row, bottom_row);
            int cell_index = i * N + j;
            if (my_grid[cell_index] == ALIVE)
            {
                new_grid[cell_index] = (alive_neighbors == 2 || alive_neighbors == 3) ? ALIVE : DEAD;
            }
            else
            {
                new_grid[cell_index] = (alive_neighbors == 3) ? ALIVE : DEAD;
            }
            if (new_grid[cell_index] == ALIVE)
            {
                localLiveCells++;
            }
        }
    }
    return localLiveCells;
}

int count_alive_neighbors(int *grid, int x, int y, int rows, int cols, int rank, int size, int *top_row, int *bottom_row)
{
    int alive_neighbors = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue; // Skip the cell itself
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols)
            {
                alive_neighbors += grid[nx * cols + ny];
            }
            else if (nx == -1 && rank > 0)
            { // top boundary
                alive_neighbors += top_row[ny];
            }
            else if (nx == rows && rank < size - 1)
            { // bottom boundary
                alive_neighbors += bottom_row[ny];
            }
        }
    }
    return alive_neighbors;
}

int count_live_cells(int *grid, int rows, int cols)
{
    int count = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (grid[i * cols + j] == ALIVE)
            {
                count++;
            }
        }
    }
    return count;
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

void process_args(int argc, char *argv[])
{
    int c = 0;

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
            width = atoi(optarg);
            // game_board_width_CPU = width;
            break;
        case 'm':
            height = atoi(optarg);
            // game_board_height_CPU = height;
            break;
        case 'i':
            i = atoi(optarg);
            // iterations = i;
            break;
        case 'o':
            o = atoi(optarg);
            // oscillator = o;
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

void print_board(int iter, int my_rows, int *my_grid)
{

    fprintf(stderr, "Round: %d\n", iter);

    for (int i = 0; i < my_rows; i++)
    { // going through grid
        for (int j = 0; j < N; j++)
        {
            if (my_grid[i * N + j] == 1)
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

void initialize_my_grid(int *grid, int rows, int cols, int rank, int size, int oscillatorFlag)
{

    if (oscillatorFlag == 1)
    {
        srand(time(NULL) + rank);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                grid[i * cols + j] = DEAD;
            }
        }

        int center_row_global = N / 2;
        int first_row_global = rank * (N / size);

        if (center_row_global >= first_row_global && center_row_global < first_row_global + rows)
        {
            int center_row_local = center_row_global - first_row_global;
            int center_col = N / 2;
            grid[center_row_local * cols + center_col - 1] = ALIVE;
            grid[center_row_local * cols + center_col] = ALIVE;
            grid[center_row_local * cols + center_col + 1] = ALIVE;
        }
    }
    else
    {
        srand(time(NULL) + rank);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                grid[i * cols + j] = rand() % 2;
            }
        }
    }
}