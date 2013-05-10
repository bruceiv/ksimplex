#pragma once

/**
 * Simple IO functions for LRS
 *
 * @author Aaron Moss
 */

#include <iostream>
#include <sstream>

#include <gmpxx.h>

#include "ksimplex.hpp"

#include "lrs/lrs.hpp"
#include "lrs/clrs.hpp"
#include "lrs/matrix.hpp"

namespace ksimplex {

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
std::string hex(const lrs::val_t& x) {
	std::stringstream s;
	s << std::hex << mpz_class(x);
	return s.str();
}

/**
 * Prints a matrix.
 * 
 * @param m			The vector to print from
 * @param n			The number of rows in the matrix (not counting objective)
 * @param d			The number of columns in the matrix (not counting constant)
 * @param out		The stream to write to
 */
void printMatrix(const lrs::vector_mpz& m, u32 n, u32 d, std::ostream& out) {
	for (u32 i = 0; i <= n; ++i) {
		for (u32 j = 0; j <= d; ++j) {
			out << " " << hex(m[1 + i*(d+1) + j]);
		}
		out << std::endl;
	}
}

} /* namespace ksimplex */
