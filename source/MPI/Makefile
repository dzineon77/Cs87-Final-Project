#
# makefile for golMPI
#
# make: builds oddevensort
# make clean: remove all compled files
# make questions: build .pdf from questions.tex 
#


CC = mpicc
CFLAGS = -g -Wall -Wvla -Werror -Wno-error=unused-variable -Wno-error=unused-but-set-variable

LIBDIR = -L./
INCDIR = -I./

TARGET = golMPI

.PHONY: clean

all: $(TARGET)

$(TARGET): $(TARGET).c 
	$(CC) $(INDIRS) $(CFLAGS) -o $(TARGET) $(TARGET).c $(LIBS) 

clean:
	$(RM) $(TARGET) *.o
