/**
 * LRS-based test driver for the KSimplex project.
 * I/O format is compatible with the usual driver, but this version uses LRS as a backend.
 * 
 * @author Aaron Moss
 */

#include <iostream>
#include <string>

#include <gmpxx.h>

#include "lrs_io.hpp"
#include "lrs_tableau.hpp"
#include "simplex.hpp"
#include "timing.hpp"

#include "lrs/lrs.hpp"
#include "lrs/clrs.hpp"
#include "lrs/cobasis.hpp"
#include "lrs/matrix.hpp"

using namespace ksimplex;

int main(int argc, char** argv) {
	// Read in size and dimension of the the problem
	u32 n, d;
	std::cin >> n;
	std::cin >> d;
	
	// Read determinant
	lrs::val_t det;
	lrs_alloc_mp(det);
	parseHex(std::cin, det);
	
	// Read basis values
	lrs::ind* bas = new lrs::ind[n+1];
	for (u32 i = 0; i <= n; ++i) std::cin >> bas[i];
	
	// Read in matrix, grab objective row first
	lrs::vector_mpz& mat = *parseLrsHex(std::cin, n, d);
	
	// Create LRS instance initialized to the given determinant and basis, and set objective
	lrs::lrs& l = *new lrs::lrs(mat, n, d, lrs::index_set(mat.size()+1), det, bas);
	
	// Construct tableau
	lrs_tableau tab(l, n, d);
	
	// Print initial tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Run simplex algorithm
	u32 pivot_count = 0;
	timer start = now();
	ksimplex::pivot p = simplexSolve(tab, &pivot_count, std::cout);
/*	// Get first pivot
	ksimplex::pivot p = tab.ratioTest();
	// Pivot as long as more pivots exist
	while ( p != tableau_optimal && p != tableau_unbounded ) {
		std::cout << "(" << p.leave << "," << p.enter << ")" << std::endl;
		tab.doPivot(p.enter, p.leave);
		++pivot_count;
		printMatrix(tab.mat(), n, d, std::cout);
		p = tab.ratioTest();
	}
*/	
	timer end = now();
	
	std::string max;
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
		
		// set maximum
		mpq_class opt;
		opt.get_num() = mpz_class(tab.obj());
		opt.get_den() = mpz_class(tab.det());
		opt.canonicalize();
		max = opt.get_str();
	} else if ( p == tableau_unbounded ) {
		std::cout << "tableau: UNBOUNDED" << std::endl;
		max = "UNBOUNDED";
	}
	
	// Print final tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Print summary information
	std::cout << "\nn:        " << n
	          << "\nd:        " << d
	          << "\npivots:   " << pivot_count
	          << "\noptimal:  " << max
	          << "\ntime(ms): " << ms_between(start, end) << std::endl;
	
	//Cleanup
	delete &l;
	delete &mat;
	delete[] bas;
	lrs_clear_mp(det);
	
	return 0;
}

