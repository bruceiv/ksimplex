# Copyright 2013 Aaron Moss
#
# This file is part of KSimplex.
#
# KSimplex is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published 
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# KSimplex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with KSimplex.  If not, see <https://www.gnu.org/licenses/>.

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
