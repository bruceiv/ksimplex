# Makefile for the KSimplex project.
# 
# @author Aaron Moss

# compiler flags
CXXFLAGS = -O2 -Wall -Wno-unused-function
# LRS compiler flags
LRSCXXFLAGS = $(CXXFLAGS) -DTIMES -DGMP -DLRS_QUIET
# CUDA compiler flags
CUDAFLAGS = -O2 -arch=compute_20 -code=sm_21
# Debug-mode CUDA compiler flags
CUDADEBUGFLAGS = -g -G -DDEBUG_CUDA -arch=compute_20 -code=sm_21 \
	--ptxas-options="-v"

# Linker flags
LDFLAGS = 
#LRS linker flags
LRSLDFLAGS = $(LDFLAGS) -Llrs -llrs -lgmpxx -lgmp

# object files to include in this executable
OBJS = 

# rules for constructions of objects from sources
.cpp.o:  
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< $(LDFLAGS)

.PHONY:  clean lrs

hksimplex:  kmp_tableau.hpp ksimplex.hpp simplex.hpp kilomp/kilomp.cuh
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o hksimplex ksimplex.cpp $(LDFLAGS)

lrspp:  lrspp.cpp lrs ksimplex.hpp
	$(CXX) $(CPPFLAGS) $(LRSCXXFLAGS) -o lrspp lrspp.cpp $(LRSLDFLAGS)

lrs:  
	cd lrs && make

tarball:  
	cd .. && tar -zcvf ksimplex.tar.gz --exclude ksimplex/.git --exclude ksimplex/.gitignore ksimplex/

clean:  
	-rm $(OBJS) sump zsump
