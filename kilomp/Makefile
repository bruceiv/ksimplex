# compiler flags
CXXFLAGS = -O2 -Wall
CUDAFLAGS = -O2 -arch=compute_20 -code=sm_21
# Linker flags
LDFLAGS = 
GMPLDFLAGS = $(LDFLAGS) -lgmp

fixed_width.hpp:  fixed_width.cpp
	${CXX} -o fixed_width fixed_width.cpp
	./fixed_width
	rm fixed_width

calc:  calc.cpp kilomp.cuh fixed_width.hpp
	${CXX} ${CXXFLAGS} -o calc calc.cpp ${LDFLAGS}

test:  test.cpp kilomp.cuh fixed_width.hpp kmp_tests.txt
	${CXX} ${CXXFLAGS} -o test test.cpp ${GMPLDFLAGS}
	./test < kmp_tests.txt
	rm test

test_cuda:  test.cu kilomp.cuh kilomp_cuda.cuh fixed_width.hpp kmp_tests.txt
	nvcc ${CUDAFLAGS} -o test_cuda test.cu ${GMPLDFLAGS}
	./test_cuda < kmp_tests.txt
	rm test_cuda

fuzz:  fuzz.cpp
	${CXX} ${CXXFLAGS} -o fuzz fuzz.cpp ${GMPLDFLAGS}

