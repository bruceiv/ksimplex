# compiler flags
CXXFLAGS = -O2 -Wall
# Linker flags
LDFLAGS = 

fixed_width.hpp:  fixed_width.cpp
	${CXX} -o fixed_width fixed_width.cpp
	./fixed_width
	rm fixed_width

calc:  calc.cpp kilomp.cuh fixed_width.hpp
	${CXX} ${CXXFLAGS} -o calc calc.cpp ${LDFLAGS}
