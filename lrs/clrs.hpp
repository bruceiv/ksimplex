#ifndef _CLRS_HPP_
#define _CLRS_HPP_

/**
 * C++ wrapper for LRS
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

#include "lrslib.h"
#include "lrsgmp.h"

/* Remove LRS macros from the global namespace */
#undef addint
#undef changesign
#undef copy
#undef decint
#undef divint
#undef exactdivint
#undef getfactorial
#undef greater
#undef gcd
#undef itomp
#undef mptoi
#undef mptodouble
#undef mulint
#undef one
#undef negative
#undef normalize
#undef positive
#undef sign
#undef subint
#undef zero

//need to declare these again, because they don't link properly otherwise
lrs_mp_matrix lrs_alloc_mp_matrix(long,long);
void lrs_clear_mp_matrix(lrs_mp_matrix,long,long);

namespace lrs {
	
	/* Bring these types into the wrapper namespace to C++-ize them */
	/** LRS lib scalar type. */
	typedef
		lrs_mp
		val_t;
	/** LRS lib vector type.
	 *  Application of the indexing operator yeilds a val_t.
	 */
	typedef 
		lrs_mp_vector
		vector_t;
	/** LRS lib matrix type.
	 *  Application of the indexing operator yeilds a vector_t.
	 */
	typedef
		lrs_mp_matrix
		matrix_t;
	/** Index type for LRS vector / matrix. */
	typedef
		long
		ind;
	/** Unsigned index type. Used to better interoperate with the unsigned 
	 *  indices of boost::dynamic_bitset
	 */
	typedef 
		unsigned long 
		uind;
	
	/* 
	 * Move LRS macros into lrs namespace 
	 */
	
	/** c = a + b */
	static void addint(val_t& a, val_t& b, val_t& c) { mpz_add(c, a, b); }
	/** a = -a */
	static void changesign(val_t& a) { mpz_neg(a, a); }
	/** dst = src */
	static void copy(val_t& dst, val_t& src) { mpz_set(dst, src); }
	/** a -= b */
	static void decint(val_t& a, val_t& b) { mpz_sub(a, a, b); }
	/** c = a / b  also  a %= b */
	static void divint(val_t& a, val_t& b, val_t& c) 
		{ mpz_tdiv_qr(c, a, a, b); }
	/** c = a / b (precondition: known there is no remainder) */
	static void exactdivint(val_t& a, val_t& b, val_t& c)
		{ mpz_divexact(c, a, b); }
	/** a = b! */
	static void getfactorial(val_t& a, unsigned long b) { mpz_fac_ui(a, b); }
	/** a > b */
	static bool greater(val_t& a, val_t& b) { return mpz_cmp(a, b) > 0; }
	/** a = gcd(a, b) */
	static void gcd(val_t& a, val_t& b) { mpz_gcd(a, a, b); }
	/** a = (val_t)in */
	static void itomp(long in, val_t& a) { mpz_set_si(a, in); }
	/** (long)a */
	static long mptoi(val_t& a) { return mpz_get_si(a); }
	/** (double)a */
	static double mptodouble(val_t& a) { return mpz_get_d(a); }
	/** c = a * b */
	static void mulint(val_t& a, val_t& b, val_t& c) { mpz_mul(c, a, b); }
	/** a == 1 */
	static bool one(val_t& a) { return mpz_cmp_si(a, 1L) == 0; }
	/** a < 0 */
	static bool negative(val_t& a) { return mpz_sgn(a) < 0; }
	/** nop */
	static void normalize(val_t& a) { }
	/** a > 0 */
	static bool positive(val_t& a) { return mpz_sgn(a) > 0; }
	/** a >= 0 ? 1 : -1 */
	static int sign(val_t& a) { return mpz_sgn(a) >= 0 ? 1 : -1; }
	/** c = a - b */
	static void subint(val_t& a, val_t& b, val_t& c) { mpz_sub(c, a, b); }
	/** a == 0 */
	static bool zero(val_t& a) { return mpz_sgn(a) == 0; }
	
} /* namespace lrs */

#endif /* _CLRS_HPP_ */