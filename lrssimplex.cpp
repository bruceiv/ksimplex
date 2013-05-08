/**
 * LRS-based test driver for the KSimplex project.
 * I/O format is compatible with the usual driver, but this version uses LRS as a backend.
 * 
 * @author Aaron Moss
 */

#include <iostream>
#include <string>

#include "lrs_io.hpp"
#include "lrs_tableau.hpp"
#include "simplex.hpp"

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
	
	// Ignore determinant
	std::string det;
	std::cin >> det;
	
	// Read in matrix, grab objective row first
	lrs::vector_mpq& obj = *parseVector(std::cin, d);
	lrs::matrix_mpq& mat = *parseMatrix(std::cin, n, d);
	
	// Create LRS instance for pre-processing, and move it to an initial basis
	lrs::lrs& l = *new lrs::lrs(mat, lrs::index_set(mat.size()+1));
	l.getFirstBasis();
	
	// Set objective into LRS instance (if you do this before initial basis, LRS solves the 
	// LP itself, which defeats the point)
	l.setLPObj(obj);
	
	// Construct tableau
	lrs_tableau tab(l, n, d);
	
	// Print initial tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Run simplex algorithm
	u32 pivot_count = 0;
	ksimplex::pivot p = simplexSolve(tab, &pivot_count, std::cout);
	
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
	} else if ( p == tableau_unbounded ) {
		std::cout << "tableau: UNBOUNDED" << std::endl;
	}
	
	// Print final tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	//Cleanup
	delete &l;
	delete &mat;
	delete &obj;
	
	return 0;
}

