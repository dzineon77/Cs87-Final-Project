# This Makefile builds a pthreads gol program that uses ParaVis library
# (updated 2021)
#
# make: builds gol executable
# make report: builds report.pdf
# make clean: clean up built stuff
#
C = gcc
C++ = g++

# two different versions of CFLAGS (uncomment the one you want to use)
# for debugging:
CFLAGS = -g -Wall -Wvla -Werror -Wno-error=unused-variable -Werror=vla
# for faster code build (with optimization on):
# CFLAGS = -02 -Wall -Wvla -Werror -Wno-error=unused-variable

#qtvis include path
INCLUDEDIR = -I/usr/local/include/qtvis

#qt5 include path
QTINCDIR = -I/usr/include/x86_64-linux-gnu/qt5

QTINCLUDES = $(QTINCDIR) \
	     $(QTINCDIR)/QtOpenGL \
	     $(QTINCDIR)/QtWidgets \
	     $(QTINCDIR)/QtGui \
	     $(QTINCDIR)/QtCore
DEFINES = -DQT_CORE_LIB -DQT_GUI_LIB -DQT_OPENGL_LIB -DQT_WIDGETS_LIB
OPTIONS = -fPIC
LIBS = $(LIBDIR) -lqtvis \
       -lQt5OpenGL -lQt5Widgets -lQt5Gui -lQt5Core -lGLX \
			 -lOpenGL -lpthread

MAINPROG=gol

all: $(MAINPROG)

#linking with link path and libs
$(MAINPROG): $(MAINPROG).o
	$(C++)  -o $(MAINPROG) \
	   $(MAINPROG).o $(LIBS)

#build the Qt5 side with no CUDA code/compiler
$(MAINPROG).o: $(MAINPROG).c colors.h
	$(CC) $(CFLAGS) $(QTINCLUDES) $(INCLUDEDIR)\
		$(OPTIONS) -c $(MAINPROG).c

report: report.tex
	pdflatex report.tex

clean:
	$(RM) $(MAINPROG) *.o *.pdf *.aux *.log *.out

