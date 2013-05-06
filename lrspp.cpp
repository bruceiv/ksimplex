/** 
 * LRS-based preprocessor to format input nicely for KSimplex.
 * A temporary measure which allows the KSimplex project to focus on the core of the simplex 
 * algorithm, rather than the neccessary pre-processing to reach an initial feasible basis.
 * 
 * @author Aaron Moss
 */

#include <iostream>
#include <sstream>
#include <string>

#include <gmpxx.h>

#include "ksimplex.hpp"

#include "lrs/lrs.hpp"
#include "lrs/clrs.hpp"
#include "lrs/cobasis.hpp"
#include "lrs/matrix.hpp"

using namespace ksimplex;

/** 
 * Reads a vector in LRS-ish format from the given input stream.
 * The first value is the constant value, followed by d coefficient values.
 * 
 * @param in		The input stream to read from
 * @param d			The number of coefficient values
 * @return the vector read from the stream (should be freed with delete)
 */
lrs::vector_mpq* parseVector(std::istream& in, u32 d) {
	//create new matrix and load data
	lrs::vector_mpq* v = new lrs::vector_mpq(d+1);
	for (u32 j = 0; j <= d; ++j) {
		in >> (*v)[j];
		(*v)[j].canonicalize();
	}
	
	return v;
}

/** 
 * Reads a matrix in LRS-ish format from the given input stream.
 * The values follow the following format, where the first column is the constant values:
 * \< n * d+1 whitespace-delimited data values \>
 * 
 * @param in		The input stream to read from
 * @param n			The number of rows
 * @param d			The number of columns (excluding the constant column)
 * @return the matrix read from the stream (should be freed with delete)
 */
lrs::matrix_mpq* parseMatrix(std::istream& in, u32 n, u32 d) {
	//create new matrix and load data
	lrs::matrix_mpq* m = new lrs::matrix_mpq(n, d+1);
	for (u32 i = 0; i < n; ++i) {
		for (u32 j = 0; j <= d; ++j) {
			in >> m->elem(i, j);
			m->elem(i, j).canonicalize();
		}
	}

	return m;
}

/** Prints a LRS value in hex, compatible with KSimplex input. */
std::string hex(lrs::val_t& x) {
	std::stringstream s;
	s << std::hex << mpz_class(x);
	return s.str();
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
	
	// Create LRS instance for pre-processing, and move it to an initial basis
	lrs::lrs& l = *new lrs::lrs(mat, lrs::index_set(mat.size()+1));
	l.getFirstBasis();
	
	// Set objective into LRS instance (if you do this before initial basis, LRS solves the 
	// LP itself, which defeats the point)
	l.setLPObj(obj);
	
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
