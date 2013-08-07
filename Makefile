# Makefile for the KSimplex project.
# 
# @author Aaron Moss

CC = gcc
CXX = g++
# compiler flags
CXXFLAGS = -O2 -Wall -Wno-unused-function
#CXXFLAGS = -O0 -ggdb -Wall -Wno-unused-function
# LRS compiler flags
LRSCXXFLAGS = $(CXXFLAGS) -DTIMES -DGMP -DLRS_QUIET
# CUDA compiler flags
CUDAFLAGS = -O2 -arch=compute_20 -code=sm_21
#CUDAFLAGS = -g -G -DDEBUG_CUDA -arch=compute_20 -code=sm_21
#	--ptxas-options="-v"

# Linker flags
LDFLAGS = 
#LRS linker flags
LRSLDFLAGS = $(LDFLAGS) -Llrs -llrs -lgmpxx -lgmp
GMPLDFLAGS = $(LDFLAGS) -lgmp

# object files to include in this executable
OBJS = 

# rules for constructions of objects from sources
.cpp.o:  
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(LDFLAGS)

.PHONY:  clean lrs

hksimplex:  ksimplex.cpp kmp_tableau.hpp ksimplex.hpp simplex.hpp kilomp/kilomp.cuh
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o hksimplex ksimplex.cpp $(LDFLAGS) -lgmpxx -lgmp

lrssimplex: lrssimplex.cpp lrs_io.hpp lrs_tableau.hpp ksimplex.hpp simplex.hpp lrs
	$(CXX) $(CPPFLAGS) $(LRSCXXFLAGS) -o lrssimplex lrssimplex.cpp $(LRSLDFLAGS)

gmpsimplex: gmpsimplex.cpp gmp_tableau.hpp ksimplex.hpp simplex.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o gmpsimplex gmpsimplex.cpp $(GMPLDFLAGS)

dksimplex:  cudasimplex.cu cuda_tableau.cuh ksimplex.hpp simplex.hpp kilomp/kilomp.cuh kilomp/kilomp_cuda.cuh
	nvcc $(CUDAFLAGS) -o dksimplex cudasimplex.cu $(LDFLAGS) -lgmp

lrspp:  lrspp.cpp lrs_io.hpp lrs ksimplex.hpp
	$(CXX) $(CPPFLAGS) $(LRSCXXFLAGS) -o lrspp lrspp.cpp $(LRSLDFLAGS)

lrs:  
	cd lrs && make

tarball:  
	cd .. && tar -zcvf ksimplex.tar.gz --exclude ksimplex/.git --exclude ksimplex/.gitignore ksimplex/

clean:  
	-rm $(OBJS) sump zsump
