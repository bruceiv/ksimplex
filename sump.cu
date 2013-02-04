/** Main driver for the "Simplex Using Multi-Precision" (sump) project
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <iostream>
#include <sstream>

#include "sump.hpp"
#include "sump_io.hpp"
#include "simplex.hpp"
#include "simple_tableau.hpp"
#include "cuda_tableau.cuh"

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
	typedef float element;
	typedef element_traits< element >::matrix matrix;
	typedef simple_tableau< element > tableau;
	
	typedef cuda_tableau< element > gpu_tableau;
	
	std::cout << "Welcome to sump!" << std::endl << std::endl;
	
	ind n, d;
	
	// Read in size and dimension of problem
	std::cin >> n;
	std::cin >> d;
	
// 	std::cout << "n:" << n << " d:" << d << std::endl;
	
	// Read in basis
	index_list bas = readIndexList(std::cin, n);
	
	// Write basis
// 	std::cout << "basis:{";
// 	writeIndexList(std::cout, bas, n);
// 	std::cout << "}" << std::endl;
	
	// Generate cobasis
	index_list cob = allocIndexList(d);
	ind c_j = 1, b_i = 1, j = 1;
	while ( b_i <= n ) {
		while ( j < bas[b_i] && c_j <= d ) cob[c_j++] = j++;
		++b_i;
	}
	while ( j <= n+d && c_j <= d ) cob[c_j++] = j++;
	
	// Write cobasis
// 	std::cout << "cobasis:{";
// 	writeIndexList(std::cout, cob, d);
// 	std::cout << "}" << std::endl;
	
	// Read in matrix
	matrix mat = readMatrix< element >(std::cin, n, d);
	
	// Write matrix
// 	std::cout << "matrix:" << std::endl;
// 	writeMatrix< element >(std::cout, mat, n, d);
	
	// Construct tableau
	tableau tab(n, d, cob, bas, mat);
	
	// Test tableau
	std::cout << std::endl << "tableau:" << std::endl;
	printTableau< tableau >(std::cout, tab);
	
	// Construct GPU tableau
	gpu_tableau gtab(n, d, cob, bas, mat);
	
	// Test GPU tableau
	std::cout << std::endl << "gpu_tableau:" << std::endl;
	printTableau< gpu_tableau >(std::cout, gtab);
	
	// Test ratio test
// 	pivot p = tab.ratioTest();
// 	
// 	std::cout << std::endl;
// 	if ( p == tableau_optimal ) {
// 		std::cout << "tableau: OPTIMAL" << std::endl;
// 	} else if ( p == tableau_unbounded ) {
// 		std::cout << "tableau: UNBOUNDED" << std::endl;
// 	} else {
// 		std::cout << "tableau: pivot(" << p.enter << "," << p.leave << ")" 
// 				<< std::endl;
// 		
// 		// Test pivot
// 		tab.pivot(p.enter, p.leave);
// 		
// 		std::cout << "tableau:" << std::endl;
// 		printTableau< tableau >(std::cout, tab);
// 	}
// 	
// 	// Test GPU ratio test
// 	p = gtab.ratioTest();
// 	
// 	std::cout << std::endl;
// 	if ( p == tableau_optimal ) {
// 		std::cout << "gpu_tableau: OPTIMAL" << std::endl;
// 	} else if ( p == tableau_unbounded ) {
// 		std::cout << "gpu_tableau: UNBOUNDED" << std::endl;
// 	} else {
// 		std::cout << "gpu_tableau: pivot(" << p.enter << "," << p.leave << ")" 
// 				<< std::endl;
// 		
// 		// Test GPU pivot
// 		gtab.pivot(p.enter, p.leave);
// 		
// 		std::cout << "gpu_tableau:" << std::endl;
// 		printTableau< gpu_tableau >(std::cout, gtab);
// 	}
	
	// Test simplex solving
	std::cout << std::endl;
	
	pivot p = simplexSolve< tableau >(tab);
	
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
	} else if ( p == tableau_unbounded ) {
		std::cout << "tableau: UNBOUNDED" << std::endl;
	}
	
	printTableau< tableau >(std::cout, tab);
	
	// Test GPU simplex solving
	std::cout << std::endl;
	
	p = simplexSolve< gpu_tableau >(gtab);
	
	if ( p == tableau_optimal ) {
		std::cout << "gpu_tableau: OPTIMAL" << std::endl;
	} else if ( p == tableau_unbounded ) {
		std::cout << "gpu_tableau: UNBOUNDED" << std::endl;
	}
	
	printTableau< gpu_tableau >(std::cout, gtab);
	
	// Cleanup
	freeIndexList(bas);
	freeIndexList(cob);
	freeMat< element >(mat);
		
	return 0;
}
