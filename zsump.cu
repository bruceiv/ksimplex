/** Main driver for integer code in the "Simplex Using Multi-Precision" (sump) 
 *  project.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>

#include "sump.hpp"
#include "sump_io.hpp"
#include "simplex.hpp"
#include "int_tableau.hpp"
#include "chimp/chimp.cuh"

/** @return time in microseconds between the time values given */
static long us(struct timeval start, struct timeval end) {
	double d_start = start.tv_sec * 1000000 + start.tv_usec;
	double d_end = end.tv_sec * 1000000 + end.tv_usec;
	return (long)(d_end - d_start);
}

/** Simple test driver.
 *  
 *  Input (tableau file):
 *  <n> <d>
 *  <basis_element>{n}
 *  (<tableau_element>{d+1}){n+1}
 *  
 *  Arguments:
 *  [<enter> <leave>]
 */
int main(int argc, char **argv) {
	using namespace sump;
	
	// Typedefs to simplify code
	typedef chimp::chimpz element;
	typedef element_traits< element >::matrix matrix;
	typedef int_tableau< element > tableau;
	
	std::cout << "Welcome to sump!" << std::endl << std::endl;
	
	ind n, d;
	chimp::chimpz det;
	
	// Read in size and dimension of problem, as well as determinant
	std::cin >> n;
	std::cin >> d;
	std::cin >> det;
	
	//std::cout << "n:" << n << " d:" << d << " det:" << det << std::endl;
	
	// Read in basis
	index_list bas = readIndexList(std::cin, n);
	
	// Write basis
 	//std::cout << "basis:{";
 	//writeIndexList(std::cout, bas, n);
 	//std::cout << "}" << std::endl;
	
	// Generate cobasis
	index_list cob = allocIndexList(d);
	ind c_j = 1, b_i = 1, j = 1;
	while ( b_i <= n ) {
		if ( j == bas[b_i] ) { ++j; ++b_i; continue; }
		while ( j < bas[b_i] && c_j <= d ) cob[c_j++] = j++;
		++b_i;
	}
	while ( j <= n+d && c_j <= d ) cob[c_j++] = j++;
	
	// Write cobasis
 	//std::cout << "cobasis:{";
 	//writeIndexList(std::cout, cob, d);
 	//std::cout << "}" << std::endl;
	
	// Read in matrix
	matrix mat = readMatrix< element >(std::cin, n, d);
	
	// Write matrix
	//std::cout << "matrix:" << std::endl;
	//writeMatrix< element >(std::cout, mat, n, d);
	
	// Construct tableau
	struct timeval init_start_time, init_end_time;
	gettimeofday(&init_start_time, 0);
	tableau tab(n, d, cob, bas, det, mat);
	gettimeofday(&init_end_time, 0);
	
	// Test tableau
	std::cout << std::endl << "tableau:" << std::endl;
	printTableau< tableau >(std::cout, tab);
	
	// Test simplex solving
	std::cout << std::endl;
	
	ind pivot_count;
	struct timeval start_time, end_time;
	gettimeofday(&start_time, 0);
	sump::pivot p = simplexSolve< tableau >(tab, &pivot_count, std::cout);
	gettimeofday(&end_time, 0);
	
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
	} else if ( p == tableau_unbounded ) {
		std::cout << "tableau: UNBOUNDED" << std::endl;
	}
	
	printTableau< tableau >(std::cout, tab);
	
	// Print quick-format stats
	std::cout << std::endl << n << " " << d << " " 
		<< pivot_count << " " << us(init_start_time, init_end_time) << " " 
		<< us(start_time, end_time) << std::endl;
	
	// Cleanup
	freeIndexList(bas);
	freeIndexList(cob);
	freeMat(mat);
	
	return 0;
}
