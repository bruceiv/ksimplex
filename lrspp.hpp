#ifndef _SUMP_LRSPP_HPP_
#define _SUMP_LRSPP_HPP_

#include <istream>
#include <ostream>
#include <string>

#include <gmp.h>

#include "sump.hpp"
#include "int_tableau.hpp"
#include "chimp/chimp.cuh"
#include "lrs/lrs.hpp"
#include "lrs/clrs.hpp"
#include "lrs/cobasis.hpp"
#include "lrs/matrix.hpp"

#include <iostream>
#include "sump_io.hpp"

/** LRS-based pre-processor for the sump project. Can be used to properly 
 *  initialize lrs instances and tableau based on chimpz.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

namespace sump {
	
	template<>
	struct element_traits<chimp::chimpz> {
		typedef chimp::chimpz* vector;
		typedef chimp::chimpz* matrix;
		
		static const element_type elType = integral;
	};
	
	/** Reads a vector in LRS-ish format from the given input stream.
	 *  The first value is the constant value, followed by d coefficient values.
	 *  @param in		The input stream to read from
	 *  @param d		The number of coefficient values
	 *  @return the vector read from the stream (should be freed with delete)
	 */
	lrs::vector_mpq* parseVector(std::istream& in, ind d) {
		//create new matrix and load data
		lrs::vector_mpq* v = new lrs::vector_mpq(d+1);
		for (ind j = 0; j <= d; ++j) {
			in >> (*v)[j];
			(*v)[j].canonicalize();
		}
		
		return v;
	}
	
	/** Reads a matrix in LRS-ish format from the given input stream.
	 *  The values follow the following format, where the first column is the 
	 *  constant values:
	 *  \< n * d+1 whitespace-delimited data values \>
	 *  @param in		The input stream to read from
	 *  @param n		The number of rows
	 *  @param d		The number of columns (excluding the constant column)
	 *  @return the matrix read from the stream (should be freed with delete)
	 */
	lrs::matrix_mpq* parseMatrix(std::istream& in, ind n, ind d) {
		//create new matrix and load data
		lrs::matrix_mpq* m = new lrs::matrix_mpq(n, d+1);
		for (ind i = 0; i < n; ++i) {
			for (ind j = 0; j <= d; ++j) {
				in >> m->elem(i, j);
				m->elem(i, j).canonicalize();
			}
		}
		
		return m;
	}
	
	/** Prints a matrix, matching its input format.
	 *  @param out		Output stream to print to
	 *  @param m		The matrix to print
	 */
	void printMatrix(std::ostream& out, lrs::matrix_mpq& m) {
		for (ind i = 0; i < m.size(); ++i) {
			for (ind j = 0; j < m.dim(); ++j) {
				out << " " << m.elem(i,j);
			}
			out << std::endl;
		}
	}
	
	/** Constructs a new LRS instance for a given matrix
	 *  @param m		The matrix to create the LRS instance for
	 *  @return a new LRS instance wrapping the matrix. Should be freed with 
	 *  		delete.
	 */
	lrs::lrs* makeLRS(lrs::matrix_mpq& m) {
		return new lrs::lrs(m, lrs::index_set(m.size()+1));
	}
	
	/** Converts an LRS-format index_set to a sump-format index_list.
	 *  @param s		The index_set to convert
	 *  @param n		The number of elements in the set
	 *  @return the index_list corresponding to the index_set
	 */
	index_list convertIndexList(const lrs::ind* s, ind n) {
		
		index_list l = allocIndexList(n);
		for (ind i = 0; i < n; ++i) {
			l[i+1] = s[i];
		}
		return l;
	}
	
	/** Converts a LRS value to a chimpz
	 *  @param x		The LRS value to convert
	 *  @return an equivalent chimpz
	 */
	chimp::chimpz convertMpz(lrs::val_t& x) {
		std::stringstream s;
		s << std::hex << mpz_class(x);
		chimp::chimpz z(s.str());
		return z;
	}
	
	/** Converts a LRS matrix to an equivalent matrix of chimpz
	 *  @param l		The LRS instance containing the matrix
	 *  @param n		The number of constraint rows in the matrix (not 
	 *  				counting the objective row)
	 *  @param d		The number of coefficient columns in the matrix (not 
	 *  				counting the constant column)
	 *  @return the equivalent matrix of chimpz (should be freed with freeMat())
	 */
	chimp::chimpz* convertMat(lrs::lrs& l, ind n, ind d) {
		chimp::chimpz* mat = allocMat<chimp::chimpz>(n, d);
		for (ind i = 0; i <= n; ++i) {
			//handle constant term
			mat[i*(d+1)] = convertMpz(l.elem(i, d));
			//get other values
			for (ind j = 0; j < d; ++j) {
				mat[i*(d+1)+j+1] = convertMpz(l.elem(i, j));
			}
		}
		
		return mat;
	}
	
	/** Constructs an int_tableau of chimpz from the given LRS instance.
	 *  @param l		The LRS instance to construct the tableau from
	 *  @return a chimpz tableau cloning the LRS instance's essential state
	 */
	int_tableau<chimp::chimpz>* makeChimpzTableau(lrs::lrs& l) {
		
		// Read n and d from LRS instance
		ind n = l.getRealSize();
		ind d = l.getRealDim();
		
		// Get cobasis from LRS instance
		lrs::ind* cob = l.getLPCobasis();
		index_list c = convertIndexList(cob, d);
		
		// Get basis from LRS instance
		lrs::ind* bas = l.getLPBasis();
		index_list b = convertIndexList(bas, n);
		
		chimp::chimpz det = convertMpz(l.getDeterminant());
		
		chimp::chimpz* mat = convertMat(l, n, d);
		
		int_tableau<chimp::chimpz>* tab 
			= new int_tableau<chimp::chimpz>(n, d, c, b, det, mat);
		
		delete[] cob;
		delete[] bas;
		freeIndexList(c);
		freeIndexList(b);
		freeMat(mat);
		
		return tab;
	}
	
} /* namespace sump */

#endif /* _SUMP_LRSPP_HPP_ */
