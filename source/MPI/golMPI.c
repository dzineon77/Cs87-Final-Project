/*
 * Swarthmore College, CS 087 - Parallel & Distributed Computing
 *
 * Game of Life Implementation in MPI
 * Dzineon Gyaltsen, Sean Cheng, Hoang Vu, Henry Lei
 * 11/23 - 12/23
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/time.h>
#include <time.h>

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

int oscillatorFlag = 0, iterations = 0;
long long int N = 0;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // process command line arguments to set board dimensions, iteration count, and oscillator flag
    process_args(argc, argv);

    // Calculate rows per process and allocate memory for the local grid
    long long int rows_per_process = N / size;
    long long int extra_rows = N % size;
    long long int my_rows = rows_per_process + (rank < extra_rows ? 1 : 0);

    int *my_grid = (int *)malloc(my_rows * N * sizeof(int));
    int *new_grid = (int *)malloc(my_rows * N * sizeof(int));
    int *top_row = (int *)malloc(N * sizeof(int));
    int *bottom_row = (int *)malloc(N * sizeof(int));

    // Initialize the grid
    initialize_my_grid(my_grid, my_rows, N, rank, size, oscillatorFlag);

    int localLiveCells = count_live_cells(my_grid, my_rows, N);

    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("my_rank = %d\n", rank);
    // printf("my_rows = %lld\n", my_rows);
    // printf("localLiveCells = %d\n", localLiveCells);
    // MPI_Barrier(MPI_COMM_WORLD);

    int globalLiveCells;
    MPI_Reduce(&localLiveCells, &globalLiveCells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (rank == 0)
    // {
    //     printf("\nglobalLiveCells BEFORE = %d\n", globalLiveCells);
    // }

    // Main loop for Game of Life iterations
    for (int iter = 0; iter < iterations; iter++)
    {
        // Communicate boundary rows
        MPI_Status status;
        if (rank != 0)
        {
            MPI_Sendrecv(my_grid, N, MPI_INT, rank - 1, 0,
                         top_row, N, MPI_INT, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
        }
        if (rank != size - 1)
        {
            MPI_Sendrecv(my_grid + (my_rows - 1) * N, N, MPI_INT, rank + 1, 1,
                         bottom_row, N, MPI_INT, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        }
        // if (rank == 2)
        // {
        //     print_board(iter, my_rows, my_grid);
        // }

        // after boundary rows are communicated, play one round of GOL and update local live cell count
        localLiveCells = play_gol(my_grid, new_grid, my_rows, rank, size, top_row, bottom_row);

        // with new local live cell count, reduce it so that global live cell count is now accurate
        MPI_Reduce(&localLiveCells, &globalLiveCells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // Swap grids for the next iteration
        int *temp = my_grid;
        my_grid = new_grid;
        new_grid = temp;
    }

    // Cleanup and finalize MPI
    free(my_grid);
    free(new_grid);
    free(top_row);
    free(bottom_row);
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

    for (long long int i = 0; i < my_rows; i++)
    {
        for (long long int j = 0; j < N; j++)
        {
            // check neighbors according to non-torus world
            int alive_neighbors = count_alive_neighbors(my_grid, i, j, my_rows, N, rank, size, top_row, bottom_row);
            long long int cell_index = i * N + j;
            if (my_grid[cell_index] == ALIVE)
            {
                // set cell to alive or dead based on neighbors
                new_grid[cell_index] = (alive_neighbors == 2 || alive_neighbors == 3) ? ALIVE : DEAD;
            }
            else
            {
                // set cell to alive or dead based on neighbors
                new_grid[cell_index] = (alive_neighbors == 3) ? ALIVE : DEAD;
            }

            if (new_grid[cell_index] == ALIVE)
            {
                // increment local live cell count if cell is alive
                localLiveCells++;
            }
        }
    }
    return localLiveCells;
}

int count_alive_neighbors(int *grid, int x, int y, int rows, int cols, int rank, int size, int *top_row, int *bottom_row)
{
    // int offset = x * N + y;
    int alive_neighbors = 0;

    // check all neighbors
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue; // Skip the cell itself

            int nx = x + i;
            int ny = y + j;

            // if valid cell not along border, increment alive_neighbors correspondingly
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols)
            {
                alive_neighbors += grid[nx * cols + ny];
            }
            else if (nx == -1 && rank > 0)
            { // handle top boundary
                alive_neighbors += top_row[ny];
            }
            else if (nx == rows && rank < size - 1)
            { // handle bottom boundary
                alive_neighbors += bottom_row[ny];
            }
        }
    }
    return alive_neighbors;
}

int count_live_cells(int *grid, int rows, int cols)
{
    int count = 0;

    // iterate through whole (local) grid to check if cells are alive
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

void process_args(int argc, char *argv[])
{
    int c = 0, o = -1, i = -1;
    long long int n = -1;

    while (1)
    {
        c = getopt(argc, argv, "o:i:n:");

        if (c == -1)
        {
            break;
        }

        switch (c)
        {
        case 'o':
            o = atoi(optarg);
            oscillatorFlag = o;
            break;
        case 'i':
            i = atoi(optarg);
            iterations = i;
            break;
        case 'n':
            n = atoll(optarg);
            N = n;
            break;
        case ':':
            fprintf(stderr, "\n Error -%c missing arg\n", optopt);
            break;
        case '?':
            fprintf(stderr, "\n Error unknown arg -%c\n", optopt);
            break;
        }
    }
}

void print_board(int iter, int my_rows, int *my_grid)
{

    fprintf(stderr, "Round: %d\n", iter);

    for (long long int i = 0; i < my_rows; i++)
    { // going through grid
        for (long long int j = 0; j < N; j++)
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
        // first set all (local) grid cells to dead,
        for (long long int i = 0; i < rows; i++)
        {
            for (long long int j = 0; j < cols; j++)
            {
                grid[i * cols + j] = DEAD;
            }
        }

        // find the center of the global grid, as well as the first row of this local grid in
        // context of whole global grid
        long long int center_row_global = N / 2;
        long long int center_col_global = N / 2;
        long long int first_row_global = rank * (N / size);
        // printf("first_row_global = %lld\n", first_row_global);

        // only if this host's local grid is the middle slice of global grid do we set the oscillator pattern
        // cells to alive
        if (center_row_global >= first_row_global && center_row_global < first_row_global + rows)
        {
            long long int center_row_local = center_row_global - first_row_global;
            // if (rank == 0) {
            //     printf("center_row_local = %lld\n", center_row_local);
            // }
            // grid[center_row_local * cols + center_col - 1 - (10*N)] = ALIVE;
            // grid[center_row_local * cols + center_col - (10*N)] = ALIVE;
            // grid[center_row_local * cols + center_col + 1 - (10*N)] = ALIVE;
            grid[center_row_local * N + center_col_global - 1] = ALIVE;
            grid[center_row_local * N + center_col_global] = ALIVE;
            grid[center_row_local * N + center_col_global + 1] = ALIVE;
        }
    }
    else
    {
        // otherwise randomly set every cell to alive or dead
        srand(time(NULL) + rank);
        for (long long int i = 0; i < rows; i++)
        {
            for (long long int j = 0; j < cols; j++)
            {
                grid[i * cols + j] = rand() % 2;
            }
        }
    }
}