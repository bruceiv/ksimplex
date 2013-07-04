# compiler flags
CXXFLAGS = -O2 -Wall
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

