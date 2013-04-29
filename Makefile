# Makefile for the KSimplex project.
# 
# @author Aaron Moss (moss.aaron@unb.ca)

LRSFLAGS = -DTIMES -DGMP -DLRS_QUIET
# compiler flags
CXXFLAGS = -O2 -Wall
# CUDA compiler flags
CUDAFLAGS = -O2 -arch=compute_20 -code=sm_21
# Debug-mode CUDA compiler flags
CUDADEBUGFLAGS = -g -G -DDEBUG_CUDA -arch=compute_20 -code=sm_21 \
	--ptxas-options="-v"
# Linker flags
LDFLAGS = 
#LRS linker flags
LRSLDFLAGS = -Llrs -llrs -lgmpxx -lgmp

# object files to include in this executable
OBJS = 

# rules for constructions of objects from sources
.cpp.o:  
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(LDFLAGS)

.PHONY:  clean lrs

# generate main program
sump:  simplex.hpp cuda_tableau.cuh sump_cuda.cuh simple_tableau.hpp \
		sump_io.hpp sump.hpp sump.cu
	nvcc $(CUDAFLAGS) -o sump sump.cu $(LDFLAGS)

zsump:  simplex.hpp int_tableau.hpp sump_io.hpp sump.hpp zsump.cu
	nvcc $(CUDAFLAGS) -o zsump zsump.cu $(LDFLAGS)

zsump_debug:  simplex.hpp int_tableau.hpp sump_io.hpp sump.hpp zsump.cu
	nvcc $(CUDADEBUGFLAGS) -o zsump_debug zsump.cu $(LDFLAGS)

gzsump:  simplex.hpp chimpz_tableau.cuh sump_io.hpp sump.hpp gzsump.cu
	nvcc $(CUDAFLAGS) -o gzsump gzsump.cu $(LDFLAGS)

gzsump_debug:  simplex.hpp chimpz_tableau.cuh sump_io.hpp sump.hpp gzsump.cu
	nvcc $(CUDADEBUGFLAGS) -o gzsump_debug gzsump.cu $(LDFLAGS)

lrspp:  sump.hpp sump_io.hpp lrs lrspp.hpp lrspp.cu
	nvcc $(CUDAFLAGS) $(LRSFLAGS) -o lrspp lrspp.cu $(LRSLDFLAGS)

lrs:  
	cd lrs && make

tarball:  
	cd .. && tar -zcvf ksimplex.tar.gz --exclude ksimplex/.git --exclude ksimplex/.gitignore ksimplex/

clean:  
	-rm $(OBJS) sump zsump
