/**
 * GMP-based version of main driver for the KSimplex project.
 */

// Copyright 2013 Aaron Moss
//
// This file is part of KSimplex.
//
// KSimplex is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published 
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KSimplex is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KSimplex.  If not, see <https://www.gnu.org/licenses/>.

#include <iostream>
#include <string>

#include <gmpxx.h>

#include "gmp_tableau.hpp"
#include "gmpv.hpp"
#include "ksimplex.hpp"
#include "simplex.hpp"
#include "timing.hpp"

using namespace ksimplex;

/**
 * Reads a word into a vector index
 * 
 * @param m			The vector to read in to
 * @param i			The index in the vector to write the new value to
 * @param m_l		The number of elements in m
 * @param in		The stream to read from
 */
void parse(mpz_t* m, u32 i, std::istream& in) {
	// Read the next word
	std::string s;
	in >> s;
	
	// Parse the value
	mpz_set_str(m[i], s.c_str(), 16);
}

/**
 * Reads a matrix from input
 * 
 * @param m			The matrix to read in to
 * @param n			The number of rows in the matrix (not counting objective)
 * @param d			The number of columns in the matrix (not counting constant)
 * @param in		The stream to read from
 */
void parseMatrix(mpz_t* m, u32 n, u32 d, std::istream& in) {
	u32 m_l = 1 + (n+1)*(d+1);
	for (u32 i = 1; i < m_l; ++i) { parse(m, i, std::cin); }
}

/**
 * Prints a vector index.
 * 
 * @param m			The vector to print from
 * @param i			The index in the vector to print from
 * @param out		The stream to write to
 */
void print(const mpz_t* m, u32 i, std::ostream& out) {
	//allocate large enough buffer
	char buf[mpz_sizeinbase(m[i], 16)+2];
	
	//print into buffer, then stdout
	mpz_get_str(buf, 16, m[i]);
	out << buf;
}

/**
 * Prints a matrix.
 * 
 * @param m			The vector to print from
 * @param n			The number of rows in the matrix (not counting objective)
 * @param d			The number of columns in the matrix (not counting constant)
 * @param out		The stream to write to
 */
void printMatrix(const mpz_t* m, u32 n, u32 d, std::ostream& out) {
	// Print determinant
	out << "["; print(m, 0, out); out << "]" << std::endl;
	// Print matrix
	for (u32 i = 0; i <= n; ++i) {
		for (u32 j = 0; j <= d; ++j) {
			out << " ";
			print(m, 1 + i*(d+1) + j, out);
		}
		out << std::endl;
	}
}

/**
 * Simple test driver.
 * 
 * Input (tableau file):
 * <n> <d> <det>
 * <basis_element>{n}
 * (<tableau_element>{d+1}){n+1}
 */
int main(int argc, char **argv) {
	// Read in size and dimension of the the problem
	u32 n, d, m_l;
	std::cin >> n;
	std::cin >> d;
	m_l = 1 + (n+1)*(d+1);
	
	// Build appropriately large mp-vector to store values
	mpz_t* mat = init_gmpv(m_l);
	
	// Read in initial determinant
	parse(mat, 0, std::cin);
	
	// Read in basis
	u32* bas = new u32[n+1];
	for (u32 i = 0; i <= n; ++i) { std::cin >> bas[i]; }
	
	// Generate cobasis
	u32* cob = new u32[d+1];
	u32 c_j = 1, b_i = 1, j = 1;
	while ( b_i <= n ) {
		if ( j == bas[b_i] ) { ++j; ++b_i; continue; }
		while ( j < bas[b_i] && c_j <= d ) { cob[c_j] = j; ++c_j; ++j; }
		++b_i;
	}
	while ( j <= n+d && c_j <= d ) { cob[c_j] = j; ++c_j; ++j; }
	
	// Read in matrix
	parseMatrix(mat, n, d, std::cin);
	
	// Construct tableau
	gmp_tableau tab(n, d, cob, bas, mat);
	
	// Print initial tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Run simplex algorithm
	u32 pivot_count = 0;
	timer start = now();
	pivot p = simplexSolve(tab, &pivot_count, std::cout);
//	// Get first pivot
//	pivot p = tab.ratioTest();
//	// Pivot as long as more pivots exist
//	while ( p != tableau_optimal && p != tableau_unbounded ) {
//		std::cout << "(" << p.leave << "," << p.enter << ")" << std::endl;
//		tab.doPivot(p.enter, p.leave);
//		++pivot_count;
//		printMatrix(tab.mat(), n, d, std::cout);
//		p = tab.ratioTest();
//	}
	
	timer end = now();
	
	std::string max;
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
		
		// set maximum
		mpq_class opt;
		opt.get_num() = mpz_class(tab.mat()[1]);
		opt.get_den() = mpz_class(tab.mat()[0]);
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
	
	// Cleanup
	delete[] cob;
	delete[] bas;
	clear_gmpv(mat, m_l);
	
	return 0;
}
