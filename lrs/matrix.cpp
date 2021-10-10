/** 
 * Implements lrs::matrix class from matrix.hpp C++ wrapper for LRS.
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
#include <cstdlib>
#include <istream>
#include <new>
#include <ostream>
#include <stdexcept>

#include <boost/functional.hpp>
#include <boost/functional/hash.hpp>

#include <gmp.h>
#include <gmpxx.h>

#include "cobasis.hpp"
#include "matrix.hpp"


namespace lrs {
	
	////////////////////////////////////////////////////////////////////////////
	//
	// VECTOR_MPQ_BASE CLASS
	//
	////////////////////////////////////////////////////////////////////////////
	
	vector_mpq_base::vector_mpq_base( mpq_class* v, ind d ) : v(v), d(d) {}
	
	////////////////////////////////////////////////////////////////////////////
	// Element access operators
	////////////////////////////////////////////////////////////////////////////
	
	mpq_class* vector_mpq_base::begin() { return v; }
	
	mpq_class const* vector_mpq_base::begin() const { return v; }
	
	mpq_class* vector_mpq_base::end() { return v + d; }
	
	mpq_class const* vector_mpq_base::end() const { return v + d; }
	
	ind vector_mpq_base::size() const { return d; }
	
	mpq_class& vector_mpq_base::operator[] ( ind i ) { return v[i]; }
	
	const mpq_class& vector_mpq_base::operator[] ( ind i ) const 
		{ return v[i]; }
	
	////////////////////////////////////////////////////////////////////////////
	// Output operator
	////////////////////////////////////////////////////////////////////////////
	
	std::ostream& operator<< ( std::ostream& o, vector_mpq_base const& v ) {
		o << "[";
		for (ind i = 0; i < v.d; i++) {
			o << " " << v.v[i];
		}
		o << " ]";
		return o;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Comparison operators
	////////////////////////////////////////////////////////////////////////////
	
	bool operator<  ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) < 0; }
	bool operator== ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) == 0; }
	bool operator>  ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) > 0; }
	bool operator<= ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) <= 0; }
	bool operator!= ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) != 0; }
	bool operator>= ( vector_mpq_base const& a, vector_mpq_base const& b ) 
		{ return compare(a,b) >= 0; }
	
	int compare( vector_mpq_base const& a, vector_mpq_base const& b ) {
		mpq_class t; int s = 0;
		
		for (ind i = 0; i < a.d && i < b.d; ++i) {
			t = a.v[i] - b.v[i];
			s = sgn(t);
			if (s != 0) return s;
		}
		// if it reaches here, the two are lexicographically equal up to the 
		// end of the shorter vector (and s == 0)
		
		if (a.d < b.d) s = -1; else if (a.d > b.d) s = 1; /* else s = 0; */
		
		return s;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Mathematical operators
	////////////////////////////////////////////////////////////////////////////
	
	vector_mpq_base operator*= ( vector_mpq_base& v, mpq_class c ) {
		for (ind i = 0; i < v.d; ++i) v.v[i] *= c;
		
		return v;
	}
	
	vector_mpq_base operator* ( vector_mpq_base& v, mpq_class c ) 
		{ vector_mpq t(v); t *= c; return t; }
	
	vector_mpq_base operator* ( mpq_class c, vector_mpq_base& v ) 
		{ vector_mpq t(v); t *= c; return t; }
	
	mpq_class inner_prod( vector_mpq_base const& a, vector_mpq_base const& b ) {
		
		if (a.d != b.d) throw std::runtime_error(
			"Cannot take inner product of vectors of unequal size");
		
		mpq_class r, t;
		for (ind i = 0; i < a.d; ++i) {
			t = a.v[i] * b.v[i];
			r += t;
		}
		
		return r;		
	}
	
	vector_mpz vector_mpq_base::num() {
		vector_mpz r(d);
		for (ind i = 0; i < d; ++i) {
			mpz_set( r.v[i], v[i].get_num_mpz_t() ); /* r[i] = v[i].num() */
		}
		return r;
	}
	
	vector_mpz vector_mpq_base::den() {
		vector_mpz r(d);
		for (ind i = 0; i < d; ++i) {
			mpz_set( r.v[i], v[i].get_den_mpz_t() ); /* r[i] = v[i].den() */
		}
		return r;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Hasher implementation
	////////////////////////////////////////////////////////////////////////////
	
	std::size_t vector_mpq_hash::operator() (vector_mpq_base const& v) const {
		std::size_t seed = 0UL;
		for (ind i = 0; i < v.d; ++i) {
			/* combine low-order bits of numerator and denominator into 
				* hash */
			boost::hash_combine(seed, v.v[i].get_num().get_si() );
			boost::hash_combine(seed, v.v[i].get_den().get_si() );
		}
		return seed;
	}
	
	
	////////////////////////////////////////////////////////////////////////////
	//
	// VECTOR_MPQ CLASS
	//
	////////////////////////////////////////////////////////////////////////////
	
	vector_mpq::vector_mpq ( ind d ) : vector_mpq_base(new mpq_class[d], d) {}
	
	vector_mpq::vector_mpq ( vector_mpq_base const& that ) 
			: vector_mpq_base(new mpq_class[that.d], that.d) {
		for (ind i = 0; i < d; i++) v[i] = that.v[i]; 
	}
	
	vector_mpq::vector_mpq ( vector_mpq const& that ) 
			: vector_mpq_base(new mpq_class[that.d], that.d) {
		for (ind i = 0; i < d; i++) v[i] = that.v[i]; 
	}
	
	vector_mpq::vector_mpq ( matrix_row_mpq const& that ) 
			: vector_mpq_base(new mpq_class[that.d], that.d) {
		for (ind i = 0; i < d; i++) v[i] = that.v[i]; 
	}
	
	vector_mpq::vector_mpq ( vector_mpz const& that ) 
			: vector_mpq_base(new mpq_class[that.d], that.d) {
		for (ind i = 0; i < d; i++) {
			/* v[i] = that.v[i]; */
			mpq_set_z(v[i].get_mpq_t(), that.v[i]);
		}
	}
	
	vector_mpq::vector_mpq ( vector_mpz const& nums, mpz_class den ) 
			: vector_mpq_base(new mpq_class[nums.d], nums.d) {
		for (ind i = 0; i < d; i++) {
			/* v[i] = nums[i] / den; */
			mpz_set(v[i].get_num_mpz_t(), nums.v[i]);
			mpz_set(v[i].get_den_mpz_t(), den.get_mpz_t());
			v[i].canonicalize();
		}
	}
	
	vector_mpq::~vector_mpq() { delete[] v; }
	
	vector_mpq& vector_mpq::operator= ( vector_mpq_base const& that ) {
		
		if (v != that.v) {
			if ( d != that.d ) {
				delete[] v;
				d = that.d;
				v = new mpq_class[d];
			}
			
			for (ind i = 0; i < d; i++) v[i] = that.v[i];
		}
		
		return *this;
	}
	
	vector_mpq& vector_mpq::operator= ( vector_mpq const& that ) {
		
		if (v != that.v) {
			if ( d != that.d ) {
				delete[] v;
				d = that.d;
				v = new mpq_class[d];
			}
			
			for (ind i = 0; i < d; i++) v[i] = that.v[i];
		}
		
		return *this;
	}
	
	vector_mpq& vector_mpq::operator= ( matrix_row_mpq const& that ) {
		
		if (v != that.v) {
			if ( d != that.d ) {
				delete[] v;
				d = that.d;
				v = new mpq_class[d];
			}
			
			for (ind i = 0; i < d; i++) v[i] = that.v[i];
		}
		
		return *this;
	}
	
	vector_mpq& vector_mpq::operator= ( vector_mpz const& that ) {
		
		if ( d != that.d ) {
			delete[] v;
			d = that.d;
			v = new mpq_class[d];
		}
		
		for (ind i = 0; i < d; i++) {
			/* v[i] = that.v[i]; */
			mpq_set_z(v[i].get_mpq_t(), that.v[i]);
		}
		
		return *this;
	}
	
	
	////////////////////////////////////////////////////////////////////////////
	//
	// MATRIX_ROW_MPQ CLASS
	//
	////////////////////////////////////////////////////////////////////////////
	
	matrix_row_mpq::matrix_row_mpq ( mpq_class* v, size_type d ) 
			: vector_mpq_base(v, d) {}
	
	matrix_row_mpq& matrix_row_mpq::operator= ( vector_mpq_base const& that ) {
		
		if (v != that.v) for (ind i = 0; i < d; i++) v[i] = that.v[i];
		
		return *this;
	}
	
	matrix_row_mpq& matrix_row_mpq::operator= ( vector_mpq const& that ) {
		
		if (v != that.v) for (ind i = 0; i < d; i++) v[i] = that.v[i];
		
		return *this;
	}
	
	matrix_row_mpq& matrix_row_mpq::operator= ( matrix_row_mpq const& that ) {
		
		if (v != that.v) for (ind i = 0; i < d; i++) v[i] = that.v[i];
		
		return *this;
	}
	
	matrix_row_mpq& matrix_row_mpq::operator= ( vector_mpz const& that ) {
		
		for (ind i = 0; i < d; i++) {
			/* v[i] = that.v[i]; */
			mpq_set_z(v[i].get_mpq_t(), that.v[i]);
		}
		
		return *this;
	}
	
	
	////////////////////////////////////////////////////////////////////////////
	//
	// VECTOR_MPZ CLASS
	//
	////////////////////////////////////////////////////////////////////////////
	
	vector_mpz::vector_mpz ( ind d ) 
			: /* lrs_alloc_mp_vector() allocs one more than needed */ 
			v(lrs_alloc_mp_vector(d-1)), d(d) {}
	
	vector_mpz::vector_mpz ( vector_mpz const& that ) 
			: /* lrs_alloc_mp_vector() allocs one more than needed */ 
			v(lrs_alloc_mp_vector(that.d-1)), d(that.d) {
		for (ind i = 0; i < d; i++) copy(v[i], that.v[i]);
	}
	
	vector_mpz::~vector_mpz() {
		/* lrs_clear_mp_vector() clears one more than needed */
		lrs_clear_mp_vector(v, d-1);
	}
	
	vector_mpz& vector_mpz::operator= ( vector_mpz const& that ) {
		if (v != that.v) {
			lrs_clear_mp_vector(v, d);
			
			d = that.d;
			v = lrs_alloc_mp_vector(d);
			for (ind i = 0; i < d; i++) copy(v[i], that.v[i]);
		}
		return *this;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Element access operators
	////////////////////////////////////////////////////////////////////////////
	
	val_t* vector_mpz::begin() { return v; }
	
	val_t const* vector_mpz::begin() const { return v; }
	
	val_t* vector_mpz::end() { return v + d; }
	
	val_t const* vector_mpz::end() const { return v + d; }
	
	val_t& vector_mpz::operator[] ( ind i ) { return v[i]; }
	
	ind vector_mpz::size() const { return d; }
	
	const val_t& vector_mpz::operator[] ( ind i ) const { return v[i]; }
	
	////////////////////////////////////////////////////////////////////////////
	// Output operator
	////////////////////////////////////////////////////////////////////////////
	
	std::ostream& operator<< ( std::ostream& o, vector_mpz const& v ) {
		o << "[";
		for (ind i = 0; i < v.d; i++) {
			o << " " << mpz_class(v[i]);
		}
		o << " ]";
		return o;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Comparison operators
	////////////////////////////////////////////////////////////////////////////
	
	bool operator<  ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) < 0; }
	bool operator== ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) == 0; }
	bool operator>  ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) > 0; }
	bool operator<= ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) <= 0; }
	bool operator!= ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) != 0; }
	bool operator>= ( vector_mpz const& a, vector_mpz const& b ) 
		{ return compare(a,b) >= 0; }
	
	int compare( vector_mpz const& a, vector_mpz const& b ) {
		mpz_class t; int s = 0;
		
		for (ind i = 0; i < a.d && i < b.d; ++i) {
			mpz_sub(t.get_mpz_t(), a.v[i], b.v[i]);	// t = a[i] - b[i];
			s = sgn(t);
			if (s != 0) return s;
		}
		// if it reaches here, the two are lexicographically equal up to the 
		// end of the shorter vector (and s == 0)
		
		if (a.d < b.d) s = -1; else if (a.d > b.d) s = 1; /* else s = 0; */
		
		return s;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Mathematical operators
	////////////////////////////////////////////////////////////////////////////
	
	vector_mpq vector_mpz::rationalization() const {
		if ( zero(v[0]) ) {
			/* return mpq_vector equivalent to this one */
			return vector_mpq(*this);
		} else {
			vector_mpq norm(this->d);
			for (ind i = 0; i < d; i++) {
				/* norm[i] = v[i] / v[0]; */
				mpz_set(norm.v[i].get_num_mpz_t(), v[i]);
				mpz_set(norm.v[i].get_den_mpz_t(), v[0]);
				norm.v[i].canonicalize();
			}
			return norm;
		}
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Hasher implementation
	////////////////////////////////////////////////////////////////////////////
	
	std::size_t vector_mpz_hash::operator() (vector_mpz const& v) const {
		std::size_t seed = 0UL;
		mpz_class one(1);
		for (ind i = 0; i < v.d; ++i) {
			/* combine low-order bits of value into hash */
			boost::hash_combine(seed, mpz_get_si(v.v[i]) );
			/* combine low-order bits of denominator 1 into hash (for 
			 * vector_mpq_hash compatibility) */
			boost::hash_combine(seed, one.get_si() );
		}
		return seed;
	}
	
	
	////////////////////////////////////////////////////////////////////////////
	//
	// MATRIX_MPQ CLASS
	//
	////////////////////////////////////////////////////////////////////////////
	
	matrix_mpq::matrix_mpq( ind n, ind d ) 
			: m(new mpq_class[n*d]), n(n), d(d) {}
	
	matrix_mpq::matrix_mpq( matrix_mpq const& that ) 
			: m(new mpq_class[that.n*that.d]), n(that.n), d(that.d) {
		for (ind i = 0; i < n*d; i++) m[i] = that.m[i];
	}
	
	matrix_mpq::~matrix_mpq() {
		delete[] m;
	}
	
	matrix_mpq& matrix_mpq::operator= ( matrix_mpq const& that ) {
		
		if (m != that.m) {
			if ( n*d != that.n*that.d ) {
				delete[] m;
				n = that.n;
				d = that.d;
				m = new mpq_class[n*d];
			}
			
			for (ind i = 0; i < n*d; i++) m[i] = that.m[i];
		}
		
		return *this;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Element access operators
	////////////////////////////////////////////////////////////////////////////
	
	matrix_mpq::iterator matrix_mpq::begin() 
		{ return matrix_mpq::iterator(m, d); }
	
	matrix_mpq::const_iterator matrix_mpq::begin() const
		{ return matrix_mpq::const_iterator(m, d); }
	
	matrix_mpq::iterator matrix_mpq::end()
		{ return matrix_mpq::iterator(m+(n*d), d); }
	
	matrix_mpq::const_iterator matrix_mpq::end() const
		{ return matrix_mpq::const_iterator(m+(n*d), d); }
	
	ind matrix_mpq::size() const { return n; }
	
	ind matrix_mpq::dim() const { return d; }
	
	matrix_row_mpq matrix_mpq::operator[] ( ind i )
		{ return matrix_row_mpq(m+(i*d), d); }
	
	matrix_row_mpq const matrix_mpq::operator[] ( ind i ) const
		{ return matrix_row_mpq(m+(i*d), d); }
	
	matrix_row_mpq matrix_mpq::row( ind i )
		{ return matrix_row_mpq(m+(i*d), d); }
	
	matrix_row_mpq const matrix_mpq::row( ind i ) const 
		{ return matrix_row_mpq(m+(i*d), d); }
	
	mpq_class& matrix_mpq::elem( ind i, ind j ) { return m[i*d+j]; }
	
	mpq_class const& matrix_mpq::elem( ind i, ind j ) const { return m[i*d+j]; }

	
	////////////////////////////////////////////////////////////////////////////
	// Output operator
	////////////////////////////////////////////////////////////////////////////
	
	std::ostream& operator<< ( std::ostream& o, matrix_mpq const& m ) {
		o << "[";
		for (ind i = 0; i < m.n; i++) {
			o << " " << m[i];
		}
		o << " ]";
		return o;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Comparison operators
	////////////////////////////////////////////////////////////////////////////
	
	bool operator<  ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) < 0; }
	
	bool operator== ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) == 0; }
	
	bool operator>  ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) > 0; }
	
	bool operator<= ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) <= 0; }
	
	bool operator!= ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) != 0; }
	
	bool operator>= ( matrix_mpq const& a, matrix_mpq const& b )
		{ return compare(a, b) >= 0; }

	int compare( matrix_mpq const& a, matrix_mpq const& b ) {
		int s = 0;
		
		for (ind i = 0; i < a.n && i < b.n; ++i) {
			s = compare(a[i], b[i]);
			if (s != 0) return s;
		}
		// if it reaches here, the two are lexicographically equal up to the 
		// end of the shorter matrix (and s == 0)
		
		if (a.n < b.n) s = -1; else if (a.n > b.n) s = 1; /* else s = 0; */
		
		return s;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Mathematical operators
	////////////////////////////////////////////////////////////////////////////
	
	matrix_mpq abs(matrix_mpq const& m) {
		matrix_mpq r(m.n, m.d);
		
		for (ind i = 0; i < m.n*m.d; ++i) r.m[i] = abs(m.m[i]);
		
		return r;
	}
	
	matrix_mpq matrix_mpq::inner_prod_mat () {
		matrix_mpq p(n,n);
		
		mpq_class t;
		for (ind i = 0; i < n; ++i) {
			/* Optimized here: p[i][j] = p[j][i], by def'n inner product */
			for (ind j = 0; j < i; ++j) {
				t = inner_prod(row(i), row(j));
				p.elem(i,j) = t; 
				p.elem(j,i) = t;
			}
			/* Handle j = i case here */
			t = inner_prod(row(i), row(i));
			p.elem(i,i) = t;
		}
		
		return p;
	}
	
	matrix_mpq matrix_mpq::restriction(index_set s) {
		matrix_mpq r(s.count(), s.count());
		
		ind i = 0;
		for (index_set_iter iterI = lrs::begin(s); 
				iterI != lrs::end(s); ++iterI) {
			ind j = 0;
			for (index_set_iter iterJ = lrs::begin(s); 
					iterJ != lrs::end(s); ++iterJ) {
				/* correct for 1-indexed index_set_iter */
				r.elem(i,j) = elem((*iterI)-1,(*iterJ)-1);
				++j;
			}
			++i;
		}
		
		return r;
	}
	
	////////////////////////////////////////////////////////////////////////////
	// Iterator implementation
	////////////////////////////////////////////////////////////////////////////
	
	matrix_mpq::iterator::iterator( mpq_class* v, ind d ) : v(v), d(d) {}
	
	matrix_mpq::const_iterator::const_iterator( mpq_class* v, ind d ) 
			: v(v), d(d) {}
	
	matrix_mpq::const_iterator::const_iterator( 
			matrix_mpq::iterator const& that ) 
			: v(that.v), d(that.d) {}
	
	matrix_row_mpq matrix_mpq::iterator::operator* () 
		{ return matrix_row_mpq(v, d); }
	
	matrix_row_mpq const matrix_mpq::const_iterator::operator* () 
		{ return matrix_row_mpq(v, d); }
	
	matrix_mpq::iterator& matrix_mpq::iterator::operator++ () 
		{ v += d; return *this; }
	
	matrix_mpq::const_iterator& matrix_mpq::const_iterator::operator++ () 
		{ v += d; return *this; }
	
	matrix_mpq::iterator matrix_mpq::iterator::operator++ (int) {
		matrix_mpq::iterator it(*this);
		v += d;
		return it;
	}
	
	matrix_mpq::const_iterator matrix_mpq::const_iterator::operator++ (int) {
		matrix_mpq::const_iterator it(*this);
		v += d;
		return it;
	}
	
	matrix_mpq::iterator& matrix_mpq::iterator::operator-- () 
		{ v -= d; return *this; }
	
	matrix_mpq::const_iterator& matrix_mpq::const_iterator::operator-- () 
		{ v -= d; return *this; }
	
	matrix_mpq::iterator matrix_mpq::iterator::operator-- (int) {
		matrix_mpq::iterator it(*this);
		v -= d;
		return it;
	}
	
	matrix_mpq::const_iterator matrix_mpq::const_iterator::operator-- (int) {
		matrix_mpq::const_iterator it(*this);
		v -= d;
		return it;
	}
	
	matrix_mpq::iterator& matrix_mpq::iterator::operator+= ( ind n ) 
		{ v += n*d; return *this; }
	
	matrix_mpq::const_iterator& matrix_mpq::const_iterator::operator+= ( ind n ) 
		{ v += n*d; return *this; }
	
	matrix_mpq::iterator matrix_mpq::iterator::operator+ ( ind n ) const 
		{ return matrix_mpq::iterator(v+(n*d), d); }
	
	matrix_mpq::const_iterator matrix_mpq::const_iterator::operator+ ( 
			ind n ) const 
		{ return matrix_mpq::const_iterator(v+(n*d), d); }
	
	matrix_mpq::iterator& matrix_mpq::iterator::operator-= ( ind n ) 
		{ v -= n*d; return *this; }
	
	matrix_mpq::const_iterator& matrix_mpq::const_iterator::operator-= ( ind n ) 
		{ v -= n*d; return *this; }
	
	matrix_mpq::iterator matrix_mpq::iterator::operator- ( ind n ) const 
		{ return matrix_mpq::iterator(v-(n*d), d); }
	
	matrix_mpq::const_iterator matrix_mpq::const_iterator::operator- ( 
			ind n ) const 
		{ return matrix_mpq::const_iterator(v-(n*d), d); }
		
	ind operator- ( matrix_mpq::iterator const& a, 
					matrix_mpq::iterator const& b ) 
		{ return (a.v - b.v) / a.d; }
	
	ind operator- ( matrix_mpq::const_iterator const& a, 
					matrix_mpq::iterator const& b ) 
		{ return (a.v - b.v) / a.d; }
	
	ind operator- ( matrix_mpq::iterator const& a, 
					matrix_mpq::const_iterator const& b ) 
		{ return (a.v - b.v) / a.d; }
	
	ind operator- ( matrix_mpq::const_iterator const& a, 
					matrix_mpq::const_iterator const& b ) 
		{ return (a.v - b.v) / a.d; }
	
	bool operator== ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v == b.v; }
	
	bool operator== ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v == b.v; }
	
	bool operator== ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v == b.v; }
	
	bool operator== ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v == b.v; }
	
	bool operator!= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v != b.v; }
	
	bool operator!= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v != b.v; }
	
	bool operator!= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v != b.v; }
	
	bool operator!= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v != b.v; }
	
	bool operator<  ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v < b.v; }
	
	bool operator<  ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v < b.v; }
	
	bool operator<  ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v < b.v; }
	
	bool operator<  ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v < b.v; }
	
	bool operator>  ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v > b.v; }
	
	bool operator>  ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v > b.v; }
	
	bool operator>  ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v > b.v; }
	
	bool operator>  ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v > b.v; }
	
	bool operator<= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v <= b.v; }
	
	bool operator<= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v <= b.v; }
	
	bool operator<= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v <= b.v; }
	
	bool operator<= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v <= b.v; }
	
	bool operator>= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v >= b.v; }
	
	bool operator>= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::iterator const& b ) 
		{ return a.v >= b.v; }
	
	bool operator>= ( matrix_mpq::iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v >= b.v; }
	
	bool operator>= ( matrix_mpq::const_iterator const& a, 
					  matrix_mpq::const_iterator const& b ) 
		{ return a.v >= b.v; }
	
	////////////////////////////////////////////////////////////////////////////
	// Hasher implementation
	////////////////////////////////////////////////////////////////////////////
	
	std::size_t matrix_mpq_hash::operator() (matrix_mpq const& m) const {
		std::size_t seed = 0UL;
		for (ind i = 0; i < m.n*m.d; ++i) {
			/* combine low-order bits of numerator and denominator into 
			 * hash */
			boost::hash_combine(seed, m.m[i].get_num().get_si() );
			boost::hash_combine(seed, m.m[i].get_den().get_si() );
		}
		return seed;
	}
	
} /* namespace lrs */
