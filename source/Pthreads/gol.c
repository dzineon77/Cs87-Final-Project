/*
 * Swarthmore College, CS 31
 * Copyright (c) 2022 Swarthmore College Computer Science Department,
 * Swarthmore PA
 */

// Original Game of Life implementation by Joey Lukner, Daniel Naagaard,
// and Zhihan Chen
// 2021

// Modifications for MPI, CUDA, and MPI-CUDA implementations performed by
// Hoang "Tommy" Vu, Dzineon Gyaltsen, Henry Lei, Sean Cheng
// November, 2023

#include <pthreadGridVisi.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include "colors.h"

/****************** Definitions **********************/
/* Three possible modes in which the GOL simulation can run */
#define OUTPUT_NONE 0  // with no animation
#define OUTPUT_ASCII 1 // with ascii animation
#define OUTPUT_VISI 2  // with ParaVis animation

/* Used to slow down animation run modes: usleep(SLEEP_USECS);
 * Change this value to make the animation run faster or slower */
#define SLEEP_USECS 200000

/* A global variable to keep track of the number of live cells in the
 * world (this is the ONLY global variable you may use in your program) */
static int total_live = 0;

/* Pthread synchronization tools initialized here */
pthread_barrier_t printBarrier;
pthread_mutex_t mutex;

/* This struct represents all the data you need to keep track of your GOL
 * simulation.  Rather than passing individual arguments into each function,
 * we'll pass in everything in just one of these structs.
 * this is passed to play_gol, the main gol playing loop */
struct gol_data
{

  int rows;        // the row dimension
  int cols;        // the column dimension
  int iters;       // number of iterations to run the gol simulation
  int output_mode; // set to:  OUTPUT_NONE, OUTPUT_ASCII, or OUTPUT_VISI
  int num_threads;
  int partition_mode;
  int rowStart, rowFinish;
  int colStart, colFinish;

  // GOL boards
  int *initialBoard;
  int *comparisonBoard;

  pthread_t *threadIDs;
  int ID;

  int oscillator;

  /* fields used by ParaVis library (when run in OUTPUT_VISI mode). */
  visi_handle handle;
  color3 *image_buff;
};

/****************** Function Prototypes **********************/

/* the main gol game playing loop */
void *play_gol(void *args);

/* Command line input verification */
void getCmdLineOpt(struct gol_data *data, int argc, char *argv[], char **filename);

/* init gol data from the input file and run mode cmdline args */
int init_game_data_from_args(struct gol_data *data, char *argv[], int argc);

/* print board to the terminal (for OUTPUT_ASCII mode) */
void print_board(struct gol_data *data, int round);

/* update colors on board (for OUTPUT_VISI mode) */
void update_colors(struct gol_data *data);

// determines how to count neighbors
int indexer(int row, int col, struct gol_data *data);

// counts number of neighbors
int numAliveNeighbors(int row, int col, struct gol_data *data);

/* used to partition the game board based on board size and number of threads*/
void partition(struct gol_data *data);

/************ Definitions for using ParVisi library ***********/
/* initialization for the ParaVisi library (DO NOT MODIFY) */
int setup_animation(struct gol_data *data);
/* register animation with ParaVisi library (DO NOT MODIFY) */
int connect_animation(void (*applfunc)(struct gol_data *data),
                      struct gol_data *data);
/* name for visi (you may change the string value if you'd like) */
static char visi_name[] = "Game of Life!";

/**********************************************************/
int main(int argc, char *argv[])
{
  // initializing variables
  int ret;
  struct gol_data data;
  double secs;
  struct gol_data *arrayOfData;
  struct timeval start_time, stop_time;

  /* Initialize game state (all fields in data) from information
   * read from input file */
  ret = init_game_data_from_args(&data, argv, argc);
  if (ret != 0)
  {
    printf("Initialization error: file %s, mode %s\n", argv[1], argv[2]);
    exit(1);
  }
  // initializes array of pthread ids to use

  /* Creates array of structs, one for each thread. Also initializes a way
   to keep track of threads */
  arrayOfData = malloc(sizeof(struct gol_data) * data.num_threads);
  data.threadIDs = malloc(sizeof(pthread_t) * data.num_threads);
  pthread_barrier_init(&printBarrier, NULL, data.num_threads);
  pthread_mutex_init(&mutex, NULL);
  // initialization after struct has number of threads

  /* initialize ParaVisi animation (if applicable) */
  if (data.output_mode == OUTPUT_VISI)
  {
    setup_animation(&data);
  }

  if (data.output_mode == OUTPUT_NONE)
  { // run with no animation
    ret = gettimeofday(&start_time, NULL);

    if (ret == -1) // error case
    {
      free(data.initialBoard);
      free(data.comparisonBoard);
      free(data.threadIDs);
      free(arrayOfData);
      printf("Error getting time of day()\n");
      return 1;
    }
  }
  else if (data.output_mode == OUTPUT_ASCII)
  { // run with ascii animation
    ret = gettimeofday(&start_time, NULL);

    if (ret == -1) // error case
    {
      free(data.initialBoard);
      free(data.comparisonBoard);
      free(data.threadIDs);
      free(arrayOfData);
      printf("Error getting time of day()\n");
      return 1;
    }
    if (system("clear"))
    {
      perror("clear");
      exit(1);
    }
    print_board(&data, 0);
  }
  else if (data.output_mode == OUTPUT_VISI)
  { // OUTPUT_VISI: run with ParaVisi animation
    run_animation(data.handle, data.iters);
  }

  // this for loops creates the threads
  for (int i = 0; i < data.num_threads; i++)
  {
    arrayOfData[i] = data;
    arrayOfData[i].ID = i;
    pthread_create(&data.threadIDs[i], NULL, play_gol, &arrayOfData[i]);
  }

  // for loop to join the threads
  for (int i = 0; i < data.num_threads; i++)
  {
    pthread_join(data.threadIDs[i], NULL);
  }

  if (data.output_mode != OUTPUT_VISI)
  { // run when mode is "not" visualization
    ret = gettimeofday(&stop_time, NULL);
    if (ret == -1)
    { // error case
      free(data.initialBoard);
      free(data.comparisonBoard);
      free(data.threadIDs);
      free(arrayOfData);
      printf("Error getting time of day()\n");
      return 1;
    }

    // getting milisecons of starttime and endtime and subtract starttime from
    // endtime
    float milistop = (stop_time.tv_usec) * (1.0 / 1000000);
    float milistart = (start_time.tv_usec) * (1.0 / 1000000);
    secs = (stop_time.tv_sec - start_time.tv_sec) + (milistop - milistart);

    /* Print the total runtime, in seconds. */
    fprintf(stdout, "Total time: %0.3f seconds\n", secs);
    fprintf(stdout, "Number of live cells after %d rounds on a %dx%d sized board: %d\n\n",
            data.iters, data.rows, data.cols, total_live);
  }

  // clean-up memories before exit
  free(data.initialBoard);
  free(data.comparisonBoard);
  free(data.threadIDs);
  free(arrayOfData);

  pthread_barrier_destroy(&printBarrier);
  pthread_mutex_destroy(&mutex);

  return 0;
}

void getCmdLineOpt(struct gol_data *data, int argc, char *argv[], char **filename)
{

  int opt;
  int valFile = 0;
  int valOutputMode = 0;
  int totnmk = 0;

  data->num_threads = 0;
  data->output_mode = OUTPUT_ASCII;
  data->partition_mode = 0;
  data->oscillator = 0;

  while (1)
  {
    opt = getopt(argc, argv, "t:n:m:k:f:sxgch"); // the : after the option in the optstring argument means the option
    // takes an arg (":tpf:" means t and p do not take arguments, and f does)

    if (opt == -1)
    {
      if (totnmk < 3 && valFile == -1)
      {
        perror("Did not specify one (or more) of the following: \n"
               "-Number of rows\n"
               "-Number of columns\n"
               "-Number of iterations\n");
        exit(1);
      }
      if (valFile == 0)
      {
        perror("Did not specify a valid file nor a valid game board");
        exit(1);
      }
      if (data->num_threads == 0)
      {
        perror("Did not specify number of threads");
        exit(1);
      }
      break; // no more command line arguments to parse, break out of while loop
    }

    switch (opt)
    {

    case 't':
      if (atoi(optarg) < 1)
      {
        perror("Invalid number of threads");
        exit(1);
      }
      (data->num_threads) = atoi(optarg);
      break;
    case 'n':
      if (valFile > 0)
      {
        perror("Can't have both input file and custom board"); // We have already been given a file
        exit(1);
      }
      if (atoi(optarg) < 1)
      {
        perror("Invalid number of rows");
        exit(1);
      }
      valFile = -1;
      totnmk++;
      (data->rows) = atoi(optarg);
      break;
    case 'm':
      if (valFile > 0)
      {
        perror("Can't have both input file and custom board"); // We have already been given a file
        exit(1);
      }
      if (atoi(optarg) < 1)
      {
        perror("Invalid number of columns");
        exit(1);
      }
      valFile = -1;
      totnmk++;
      (data->cols) = atoi(optarg);
      break;
    case 'k':
      if (valFile > 0)
      {
        perror("Can't have both input file and custom board"); // We have already been given a file
        exit(1);
      }
      if (atoi(optarg) < 1)
      {
        perror("Invalid number of iterations");
        exit(1);
      }
      valFile = -1;
      totnmk++;
      (data->iters) = atoi(optarg);
      break;
    case 'f':
      if (valFile < 0)
      {
        perror("Can't have both input file and custom board"); // we have already been given tnmk
        exit(1);
      }
      valFile = 1;
      *filename = optarg;
      break;
    case 's':
      if (valFile > 0)
      {
        perror("Can't have both input file and custom board"); // We have already been given a file
        exit(1);
      }
      valFile = -1;
      (data->oscillator) = 1;
      break;
    case 'x':
      if (valOutputMode > 0)
      {
        perror("Can't have two Output Modes");
        exit(1);
      }
      valOutputMode = -1;
      (data->output_mode) = OUTPUT_NONE;
      break;
    case 'g':
      if (valOutputMode < 0)
      {
        perror("Can't have two Output Modes");
        exit(1);
      }
      valOutputMode = 1;
      (data->output_mode) = OUTPUT_VISI;
      break;
    case 'c':
      (data->partition_mode) = 1;
      break;
    case 'h':
      perror("./gol -t t { -n n -m m -k k [-s] | -f infile} [-x] [-c] "
             "-t  t:      number of threads"
             "-n  n:      number of rows in the game board"
             "-m  m:      number of columns in the game board"
             "-k  k:      number of iterations"
             "-s:         initialize to oscillator pattern (default is random)"
             "-f infile:  read in board config info from an input file"
             "-x:         disable ascii animation (printing out board every iteration)"
             "(default is to run with animation/printing)"
             "-c:         do column partitioning (default is row portioning)"
             "-h:         print out this help message)");
      exit(1);
      break;
    } // ... some code to handle the current option ...
  }
}

int init_game_data_from_args(struct gol_data *data, char *argv[], int argc)
{
  // initializing variables
  FILE *inFile;
  int ret;

  int x, y;
  char *fileName = NULL;

  getCmdLineOpt(data, argc, argv, &fileName);

  if (fileName != NULL)
  {

    inFile = fopen(fileName, "r");

    ret = fscanf(inFile, "%d", &(data->rows)); // capturing # of grid row
    if (ret == -1)                             // error case
    {
      printf("error reading rows\n");
      exit(1);
    }
    ret = fscanf(inFile, "%d", &(data->cols)); // capturing # of grid column

    if (ret == -1) // error case
    {
      printf("error reading cols\n");
      exit(1);
    }
    ret = fscanf(inFile, "%d", &(data->iters)); // capturing # of iterations
    if (ret == -1)                              // error case
    {
      printf("error reading iterations\n");
      exit(1);
    }

    ret = fscanf(inFile, "%d", &total_live); // capturing # of total live cells
    if (ret == -1)                           // error case
    {
      printf("error reading num live cells\n");
      exit(1);
    }
  }

  // allocating memory spaces
  data->initialBoard = malloc((sizeof(int) * (data->rows) * (data->cols)));
  data->comparisonBoard = malloc((sizeof(int) * (data->rows) * (data->cols)));

  // setting all grid to be 0
  for (int i = 0; i < (data->rows); i++)
  {
    for (int j = 0; j < (data->cols); j++)
    {
      data->initialBoard[i * (data->cols) + j] = 0;
      data->comparisonBoard[i * (data->cols) + j] = 0;
    }
  }

  // When oscillator flag is set to 0 (default), create a random pattern of
  // intially live cells on the initialBoard
  if (data->oscillator == 0 && fileName == NULL)
  {
    srand(time(NULL)); // call ONLY ONE TIME in program to seed rand num generator
    int isalive;

    for (int i = 0; i < (data->rows); i++)
    {
      for (int j = 0; j < (data->cols); j++)
      {
        isalive = rand() % 2;
        data->initialBoard[i * (data->cols) + j] = isalive;
        total_live += isalive;
        data->comparisonBoard[i * (data->cols) + j] = 0;
      }
    }
  }

  // When oscillator flag is set to 1, initialize initialBoard to a horizontal line
  // of width 3 at the center of the board
  else if (data->oscillator == 1 && fileName == NULL)
  {
    int midcol = (data->cols) / 2;
    int midrow = (data->rows) / 2;

    data->initialBoard[midrow * (data->cols) + midcol - 1] = 1;
    data->initialBoard[midrow * (data->cols) + midcol] = 1;
    data->initialBoard[midrow * (data->cols) + midcol + 1] = 1;
    total_live += 3;
  }

  // getting coordinators that are alive at the beginning from the file and
  // replacing it with 1, indicating that it is alive at the coordinate.
  if (fileName != NULL)
  {
    for (int i = 0; i < total_live; i++)
    {
      ret = fscanf(inFile, "%d %d", &x, &y);
      if (ret == -1)
      {
        printf("Improper file format with coordinates\n");
        exit(1);
      }
      data->initialBoard[x * (data->cols) + y] = 1;
      // data -> comparisonBoard[x * (data->cols) + y] = 1;
    }
  }

  // sets the number of threads to max rows/cols if the amount inputted
  // is greater than the board size
  if (data->partition_mode == 1 && data->num_threads > data->cols)
  {
    data->num_threads = data->cols;
  }
  else if (data->partition_mode == 0 && data->num_threads > data->rows)
  {
    data->num_threads = data->rows;
  }

  return 0;
}

/*these two functions calculate the number of neighbors that are alive
for a given cell. It returns the in number of alive neighbors and take
the index of the cell as well as a gol_data struct*/
int indexer(int row, int col, struct gol_data *data)
{
  if (col < 0)
  {
    col = (data->cols) - 1;
  }
  if (col > (data->cols) - 1)
  {
    col = 0;
  }
  if (row < 0)
  {
    row = (data->rows) - 1;
  }
  if (row > (data->rows) - 1)
  {
    row = 0;
  }
  return row * (data->cols) + col;
}

int numAliveNeighbors(int row, int col, struct gol_data *data)
{
  int counter = 0;

  if (row == 0 && col == 0)
  {
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
    counter += data->initialBoard[indexer((row + 1), (col) + 1, data)];
  }
  else if (row == (data->rows) - 1 && col == 0)
  {
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row - 1), (col + 1), data)];
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
  }
  else if (row == 0 && col == (data->cols) - 1)
  {
    counter += data->initialBoard[indexer((row + 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
    counter += data->initialBoard[indexer((row), (col)-1, data)];
  }
  else if (row == (data->rows) - 1 && col == (data->cols) - 1)
  {
    counter += data->initialBoard[indexer((row - 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row), (col)-1, data)];
  }
  else if (row == 0)
  {
    counter += data->initialBoard[indexer((row), (col)-1, data)];
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
    counter += data->initialBoard[indexer((row + 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
    counter += data->initialBoard[indexer((row + 1), (col) + 1, data)];
  }
  else if (row == (data->rows) - 1)
  {
    counter += data->initialBoard[indexer((row - 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row - 1), (col + 1), data)];
    counter += data->initialBoard[indexer((row), (col)-1, data)];
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
  }
  else if (col == 0)
  {
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row - 1), (col + 1), data)];
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
    counter += data->initialBoard[indexer((row + 1), (col) + 1, data)];
  }
  else if (col == (data->cols) - 1)
  {
    counter += data->initialBoard[indexer((row - 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row), (col)-1, data)];
    counter += data->initialBoard[indexer((row + 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
  }
  else
  {
    counter += data->initialBoard[indexer((row - 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row - 1), (col), data)];
    counter += data->initialBoard[indexer((row - 1), (col + 1), data)];
    counter += data->initialBoard[indexer((row), (col)-1, data)];
    counter += data->initialBoard[indexer((row), (col) + 1, data)];
    counter += data->initialBoard[indexer((row + 1), (col - 1), data)];
    counter += data->initialBoard[indexer((row + 1), (col), data)];
    counter += data->initialBoard[indexer((row + 1), (col) + 1, data)];
  }
  return counter;
}

/**********************************************************/
/* the gol application main loop function:
 *  runs rounds of GOL,
 *    * updates program state for next round (world and total_live)
 *    * performs any animation step based on the output/run mode
 *
 *   data: pointer to a struct gol_data  initialized with
 *         all GOL game playing state
 */
void *play_gol(void *args)
{
  struct gol_data *data;
  data = (struct gol_data *)args;

  partition(data);

  int neighbors; // initializing a variable

  int local_live; // initilizing local variable for living cells

  for (int k = 0; k < data->iters; k++)
  { // runs until speified # of iterations
    local_live = 0;
    for (int i = data->rowStart; i < data->rowFinish; i++)
    { // runs until # of grid rows

      for (int j = data->colStart; j < data->colFinish; j++)
      { // runs until # of grid columns

        // printf("col start %d col finish %d\n", boundaries[2], boundaries[3]);
        // getting number of alive neighbors using helper function numAliveNeighbors
        neighbors = numAliveNeighbors(i, j, data);
        if (data->initialBoard[i * (data->cols) + j] == 1)
        {
          // if specified coordinate is alive
          if (neighbors == 2 || neighbors == 3)
          { // alive case
            // stay alive
            data->comparisonBoard[i * (data->cols) + j] = 1;
          }
          else
          {                                                  // dead case
            data->comparisonBoard[i * (data->cols) + j] = 0; // change to dead.0
            local_live--;
          }
        }
        else
        { // if specified coordinate is dead/0
          if (neighbors == 3)
          {                                                  // revive case
            data->comparisonBoard[i * (data->cols) + j] = 1; // set up as alive

            local_live++;
          }
          else
          { // not a revive case
            // remains dead
            data->comparisonBoard[i * (data->cols) + j] = 0;
          }
        }
      }
    }

    pthread_mutex_lock(&mutex);
    total_live += local_live;
    pthread_mutex_unlock(&mutex);

    pthread_barrier_wait(&printBarrier);
    // case when output_mode is ASCII (1)
    if (data->output_mode == OUTPUT_ASCII && data->ID == 0)
    {
      system("clear");
      print_board(data, k + 1);
      usleep(SLEEP_USECS);
    }

    pthread_barrier_wait(&printBarrier);

    if (data->output_mode == OUTPUT_VISI)
    // case output_mode is OUTPUT_VISI(2)
    {
      update_colors(data);
      draw_ready(data->handle);
      usleep(SLEEP_USECS);
    }

    int *temp = data->initialBoard;
    data->initialBoard = data->comparisonBoard;
    data->comparisonBoard = temp;

    pthread_barrier_wait(&printBarrier);

  } // end of one full iteration here
  pthread_barrier_wait(&printBarrier);
  return NULL;
} // end of function

/**********************************************************/
/* Print the board to the terminal.
 *   data: gol game specific data
 *   round: the current round number
 */
void print_board(struct gol_data *data, int round)
{

  int i, j;
  /* Print the round number. */
  fprintf(stderr, "Round: %d\n", round);
  for (i = 0; i < (data->rows); i++)
  { // going through grid
    for (j = 0; j < (data->cols); j++)
    {
      if ((data->initialBoard[i * (data->cols) + j]) == 1)
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
  /* Print the total number of live cells. */
  fprintf(stderr, "Live cells: %d\n\n", total_live);
  usleep(SLEEP_USECS * 2);
}

void update_colors(struct gol_data *data)
{ // referred from inlab

  int i, j, r, c, buff_i;
  color3 *buff;

  buff = data->image_buff; // just for readability
  r = data->rows;
  c = data->cols;

  for (i = data->rowStart; i < data->rowFinish; i++)
  {
    for (j = data->colStart; j < data->colFinish; j++)
    {

      // translate row index to y-coordinate value
      // in the image buffer, r,c=0,0 is assumed to be the _lower_ left
      // in the grid, r,c=0,0 is _upper_ left.
      buff_i = (r - (i + 1)) * c + j;

      // update animation buffer
      if (data->comparisonBoard[indexer(i, j, data)] == 0)
      {
        buff[buff_i] = colors[(data->ID) % 8];
      }
      else
      {
        buff[buff_i] = c3_black;
      }
    }
  }
}

/**********************************************************/
/***** START: DO NOT MODIFY THIS CODE *****/
/* initialize ParaVisi animation */
int setup_animation(struct gol_data *data)
{
  /* connect handle to the animation */
  int num_threads = data->num_threads;
  data->handle = init_pthread_animation(num_threads, data->rows,
                                        data->cols, visi_name);
  if (data->handle == NULL)
  {
    printf("ERROR init_pthread_animation\n");
    exit(1);
  }
  // get the animation buffer
  data->image_buff = get_animation_buffer(data->handle);
  if (data->image_buff == NULL)
  {
    printf("ERROR get_animation_buffer returned NULL\n");
    exit(1);
  }
  return 0;
}

/* sequential wrapper functions around ParaVis library functions */
void (*mainloop)(struct gol_data *data);

void *seq_do_something(void *args)
{
  mainloop((struct gol_data *)args);
  return 0;
}

int connect_animation(void (*applfunc)(struct gol_data *data), struct gol_data *data)
{
  pthread_t pid;

  mainloop = applfunc;
  if (pthread_create(&pid, NULL, seq_do_something, (void *)data))
  {
    printf("pthread_created failed\n");
    return 1;
  }
  return 0;
}

// row partitioning is 0 column partitioning is 1
void partition(struct gol_data *data)
{
  int quotient;
  int remainder;

  // printf("%d\n",tid );
  if (data->partition_mode == 0)
  {
    // colstart
    data->colStart = 0;
    // colfinish
    data->colFinish = (data->cols);

    // finds more data for offset
    quotient = data->rows / data->num_threads;
    remainder = data->rows % data->num_threads;

    // if bigger chunk

    if (data->ID < remainder)
    {
      // set rowstart
      data->rowStart = data->ID * (quotient + 1);

      // set rowfinish
      data->rowFinish = data->rowStart + quotient + 1;
    }
    // if smaller chunk
    else
    {
      // set rowstart
      data->rowStart = remainder + (data->ID * quotient);

      // set rowfinish
      data->rowFinish = data->rowStart + quotient;
    }
  }
  // column partitioning
  else if (data->partition_mode == 1)
  {
    // rowstart
    data->rowStart = 0;
    // rowFinish
    data->rowFinish = (data->rows);

    // finds more data for offset
    quotient = data->cols / data->num_threads;
    remainder = data->cols % data->num_threads;

    // if bigger chunk
    if (data->ID < remainder)
    {
      // set colstart
      data->colStart = data->ID * (quotient + 1);

      // set colfinish
      data->colFinish = data->colStart + quotient + 1;
    }
    // if smaller chunk
    else
    {
      // set colstart
      data->colStart = remainder + (data->ID * quotient);

      // set colfinish
      data->colFinish = data->colStart + quotient;
    }
  }
  // printf("exiting now thread %d\n", tid);
}

/***** END: DO NOT MODIFY THIS CODE *****/
/******************************************************/
