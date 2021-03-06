/** 
 * LRS-based preprocessor to format input nicely for KSimplex.
 * A temporary measure which allows the KSimplex project to focus on the core of the simplex 
 * algorithm, rather than the neccessary pre-processing to reach an initial feasible basis.
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
#include <sstream>
#include <string>

#include <gmpxx.h>

#include "ksimplex.hpp"
#include "lrs_io.hpp"

#include "lrs/lrs.hpp"
#include "lrs/clrs.hpp"
#include "lrs/cobasis.hpp"
#include "lrs/matrix.hpp"

using namespace ksimplex;

/** 
 * Reduces the rows in an LRS instance to lowest terms.
 */
void toLowestTerms(lrs::lrs& l, u32 n, u32 d) {
	// Initialize GCD to determinant
	mpz_t g;
	mpz_init(g);
	
	// Calculate GCD
	for (u32 i = 0; i <= n; ++i) {
		mpz_set(g, l.elem(i, 0));
		
		// while GCD is != 1
		for (u32 j = 1; j <= d && mpz_cmp_ui(g, 1) != 0; ++j) {
			mpz_gcd(g, g, l.elem(i, j));      // g = gcd(g, l[i,j])
		}
		
		if ( mpz_cmp_ui(g, 1) > 0 ) {  // gcd is > 1
			for (u32 j = 0; j <= d; ++j) {
				mpz_divexact(l.elem(i, j), l.elem(i, j), g);
			}
		}
	}
	
	mpz_clear(g);
}

/**
 * Finds an initial basis for a LP, and outputs in a format readable by KSimplex.
 * 
 * Input:
 * <n> <d>
 * (<tableau_element>{d+1}){n+1}
 * 
 * Output:
 * <n> <d> <det>
 * <basis_element>{n}
 * (<tableau_element>{d+1}){n+1}
 */
int main(int argc, char** argv) {
	u32 n, d;
	
	// Read in size and dimension of problem
	std::cin >> n;
	std::cin >> d;
	
	// Read in matrix, grab objective row first
	lrs::vector_mpq& obj = *parseVector(std::cin, d);
	lrs::matrix_mpq& mat = *parseMatrix(std::cin, n, d);
	
	// Create LRS instance for pre-processing, set the objective, and move it to an initial basis 
	// without solving the LP
	lrs::lrs& l = *new lrs::lrs(mat, lrs::index_set(mat.size()+1));
	l.setLPObj(obj);
	l.getFirstBasis(false);
	
	// Reduce tableau to lowest terms
//	toLowestTerms(l, n, d);
	
	// Write size and dimension of problem, as well as determinant
	std::cout << n << " " << d << " " << hex(l.getDeterminant()) << std::endl;
	
	// Write initial basis
	lrs::ind* bas = l.getLPBasis();
	std::cout << 0;
	for (u32 i = 0; i < n; ++i) { std::cout << " " << bas[i]; }
	std::cout << std::endl;
	
	// Write objective/constraint matrix
	for (u32 i = 0; i <= n; ++i) {
		// Write constant term
		std::cout << hex(l.elem(i,d));
		// Write matrix elements
		for (u32 j = 0; j < d; ++j) { std::cout << " " << hex(l.elem(i,j)); }
		std::cout << std::endl;
	}
	
	// Cleanup
	delete[] bas;
	delete &l;
	delete &mat;
	delete &obj;
	
	return 0;
}
