#ifndef _LRS_MATRIX_HPP_
#define _LRS_MATRIX_HPP_

/** 
 * C++ wrapper for LRS matrix.
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

#include <cstdlib>
#include <functional>
#include <istream>
#include <iterator>
#include <new>
#include <ostream>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include <gmp.h>
#include <gmpxx.h>

#include "clrs.hpp"
#include "cobasis.hpp"

namespace lrs {
	
	class vector_mpq;
	class matrix_row_mpq;
	class vector_mpz;
	class vector_mpq_hash;
	
	/** Wraps a multi-precision rational vector. */
	class vector_mpq_base {
	friend class vector_mpq;
	friend class matrix_row_mpq;
	friend class vector_mpq_hash;
	public:
		/* typedef's for STL compatibility */
		typedef mpq_class&			reference;
		typedef mpq_class const&	const_reference;
		typedef mpq_class* 			iterator;
		typedef mpq_class const*	const_iterator;
		typedef ind					size_type;
		typedef ind					difference_type;
		typedef mpq_class			value_type;
		typedef mpq_class*			pointer;
		typedef mpq_class const*	const_pointer;
		
		/** Gets an iterator at the first element of the vector. */
		iterator begin();
		
		/** Gets a constant iterator at the first element of the vector. */
		const_iterator begin() const;
		
		/** Gets an iterator one past the last element of the vector. */
		iterator end();
		
		/** Gets a constant iterator one past the last element of the vector. */
		const_iterator end() const;
		
		/** Gets the dimension of this vector. */
		size_type size() const;
		
		/** Indexing operator. Returns a mutable element reference.
		 *  @param i		The index of the element to return.
		 */
		reference operator[] (size_type i);
		
		/** Indexing operator. Returns a constant element reference.
		 *  @param i		The index of the element to return.
		 */
		const_reference operator[] (size_type i) const;
		
		/** Prints the vector with space-separated elements inside square 
		 *  brackets.
		 *  @param o		The output stream to print on
		 *  @param v		The vector to print
		 *  @return the same output stream
		 */
		friend std::ostream& operator<< (std::ostream& o, 
										 vector_mpq_base const& v);
		
		/** Less than. Equivalent to a.compare(b) < 0 */
		friend bool operator<  (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		/** Equal. Equivalent to a.compare(b) == 0 */
		friend bool operator== (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		/** Greater than. Equivalent to a.compare(b) > 0 */
		friend bool operator>  (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		/** Less than or equal. Equivalent to a.compare(b) <= 0 */
		friend bool operator<= (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		/** Not equal. Equivalent to a.compare(b) != 0 */
		friend bool operator!= (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		/** Greater than or equal. Equivalent to a.compare(b) >= 0 */
		friend bool operator>= (vector_mpq_base const& a, 
								vector_mpq_base const& b);
		
		/** Compares two vectors lexicographically.
		 *  @param a		the first vector
		 *  @param b		the second vector
		 *  @return -1 for a less than b, 0 for a equal to b, or 1 for a 
		 *  		greater than b.
		 */
		friend int compare(vector_mpq_base const& a, vector_mpq_base const& b);
		
		/** Multiplies v by c
		 *  @param v		the vector to multiply
		 *  @param c		the scalar to multiply v by
		 *  @return v, having been multiplied by c
		 */
		friend vector_mpq_base operator*= (vector_mpq_base& v, mpq_class c);
		
		/** Scalar multiplication of c and v
		 *  @param v		the vector to multiply
		 *  @param c		the scalar to multiply v by
		 *  @return the scalar multiplication of c and v.
		 */
		friend vector_mpq_base operator* (vector_mpq_base& v, mpq_class c);
		friend vector_mpq_base operator* (mpq_class c, vector_mpq_base& v);
		
		/** Computes the inner product of two vectors.
		 *  @param a		A vector of length d
		 *  @param b		A vector of length d
		 *  @return \f$\sum_{i=0}^{d} a_i * b_i\f$
		 *  @throws std::runtime_error on a.d != b.d
		 */
		friend mpq_class inner_prod (vector_mpq_base const& a, 
									 vector_mpq_base const& b);
		
		/** Gets the vector of numerators of this vector */
		vector_mpz num();
		/** Gets the vector of denominators of this vector */
		vector_mpz den();
		
	protected:
		
		/** Protected constructor. Must be called from subclass.
		 *  @param v		The data storage array
		 *  @param d		The length of the data
		 */
		vector_mpq_base(mpq_class* v, size_type d);
		
		/** Internal storage of vector data */
		mpq_class* v;
		/** Dimension of the vector. */
		size_type d;
	};
	
	int compare(vector_mpq_base const& a, vector_mpq_base const& b);
	mpq_class inner_prod (vector_mpq_base const& a, vector_mpq_base const& b);
	
	class vector_mpz;
	
	/** Wraps a self-allocated multi-precision rational vector. */
	class vector_mpq : public vector_mpq_base {
	friend class vector_mpz;
	public:
		
		/** Constructs a vector with the given dimension, with all elements 
		 *  initialized to zero. If called as the default constructor, creates 
		 *  an empty vector.
		 *  @param d		The dimension of the vector (default 0)
		 */
		vector_mpq(size_type d = 0);
		
		/** Copy constructor.
		 *  @param that		The vector to copy
		 */
		vector_mpq(vector_mpq_base const& that);
		
		/** Copy constructor.
		 *  @param that		The vector to copy
		 */
		vector_mpq(vector_mpq const& that);
		
		/** Copy constructor.
		 *  @param that		The vector to copy
		 */
		vector_mpq(matrix_row_mpq const& that);
		
		/** Copy constructor.
		 *  @param that		The integer vector to copy. Will initialize all 
		 * 					denominators to 1
		 */
		vector_mpq(vector_mpz const& that);
		
		/** Rationalization constructor. Will reduce all values to canonical 
		 *  form.
		 *  @param nums		The integer vector of numerators
		 *  @param den		The denominator of each value
		 */
		vector_mpq(vector_mpz const& nums, mpz_class den);
		
		/** Destructor. */
		~vector_mpq();
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one.
		 */
		vector_mpq& operator= (vector_mpq_base const& that);
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one.
		 */
		vector_mpq& operator= (vector_mpq const& that);
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one.
		 */
		vector_mpq& operator= (matrix_row_mpq const& that);
		
		/** Assignment operator.
		 *  @param that		The integer vector to copy into this one. Will 
		 * 					initialize all denominators to 1.
		 */
		vector_mpq& operator= (vector_mpz const& that);
		
	};
	
	class matrix_mpq;
	
	/** Multi-precision rational vector that is a view of a matrix row */
	class matrix_row_mpq : public vector_mpq_base {
	friend class matrix_mpq;
	public:
		
		/** Creates a vector that is a view of a given matrix row, rather than 
		 *  having its own memory. */
		matrix_row_mpq(mpq_class* v, size_type d);
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one. Undefined results 
		 * 					if that.d differs from this->d.
		 */
		matrix_row_mpq& operator= (vector_mpq_base const& that);
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one. Undefined results 
		 * 					if that.d differs from this->d.
		 */
		matrix_row_mpq& operator= (vector_mpq const& that);
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one. Undefined results 
		 * 					if that.d differs from this->d.
		 */
		matrix_row_mpq& operator= (matrix_row_mpq const& that);
		
		/** Assignment operator.
		 *  @param that		The integer vector to copy into this one. Will 
		 * 					initialize all denominators to 1. Undefined results 
		 * 					if that.d differs from this->d.
		 */
		matrix_row_mpq& operator= (vector_mpz const& that);
		
	};
	
	/** Functional to hash a vector_mpq.  */
	class vector_mpq_hash 
			: public std::unary_function<vector_mpq_base, std::size_t> {
	public:
		std::size_t operator() (vector_mpq_base const& v) const;
	};
	
	class vector_mpz_hash;
	
	/** Wraps an LRS-compatible multi-precision integer vector. */
	class vector_mpz {
	//assign these friends so they can get at the v member
	friend class lrs;
	friend class vector_mpq_base;
	friend class vector_mpq;
	friend class matrix_row_mpq;
	friend class vector_mpz_hash;
	public:
		/* typedef's for STL compatibility */
		typedef val_t&			reference;
		typedef val_t const&	const_reference;
		typedef val_t* 			iterator;
		typedef val_t const*	const_iterator;
		typedef ind				size_type;
		typedef ind				difference_type;
		typedef val_t			value_type;
		typedef val_t*			pointer;
		typedef val_t const*	const_pointer;
		
		/** Constructs a vector with the given dimension, with all elements 
		 *  initialized to zero. If called as the default constructor, creates 
		 *  an empty vector.
		 *  @param d		The dimension of the vector (default 0)
		 */
		vector_mpz(size_type d = 0);
		
		/** Copy constructor.
		 *  @param that		The vector to copy
		 */
		vector_mpz(vector_mpz const& that);
		
		/** Destructs a vector with the given dimension. */
		~vector_mpz();
		
		/** Assignment operator.
		 *  @param that		The vector to copy into this one.
		 */
		vector_mpz& operator= (vector_mpz const& that);
		
		/** Gets an iterator at the first element of the vector. */
		iterator begin();
		
		/** Gets a constant iterator at the first element of the vector. */
		const_iterator begin() const;
		
		/** Gets an iterator one past the last element of the vector. */
		iterator end();
		
		/** Gets a constant iterator one past the last element of the vector. */
		const_iterator end() const;
		
		/** Gets the dimension of this vector. */
		size_type size() const;
		
		/** Indexing operator. Returns a mutable element reference.
		 *  @param i		The index of the element to return.
		 */
		reference operator[] (size_type i);
		
		/** Indexing operator. Returns a constant element reference.
		 *  @param i		The index of the element to return.
		 */
		const_reference operator[] (size_type i) const;
		
		/** Prints the vector with space-separated elements inside square 
		 *  brackets.
		 *  @param o		The output stream to print on
		 *  @param v		The vector to print
		 *  @return the same output stream
		 */
		friend std::ostream& operator<< (std::ostream& o, vector_mpz const& v);
		
		/** Less than. Equivalent to a.compare(b) < 0 */
		friend bool operator<  (vector_mpz const& a, vector_mpz const& b);
		/** Equal. Equivalent to a.compare(b) == 0 */
		friend bool operator== (vector_mpz const& a, vector_mpz const& b);
		/** Greater than. Equivalent to a.compare(b) > 0 */
		friend bool operator>  (vector_mpz const& a, vector_mpz const& b);
		/** Less than or equal. Equivalent to a.compare(b) <= 0 */
		friend bool operator<= (vector_mpz const& a, vector_mpz const& b);
		/** Not equal. Equivalent to a.compare(b) != 0 */
		friend bool operator!= (vector_mpz const& a, vector_mpz const& b);
		/** Greater than or equal. Equivalent to a.compare(b) >= 0 */
		friend bool operator>= (vector_mpz const& a, vector_mpz const& b);
		
		/** Compares two vectors lexicographically.
		 *  @param a		the first vector
		 *  @param b		the second vector
		 *  @return -1 for a less than b, 0 for a equal to b, or 1 for a 
		 *  		greater than b.
		 */
		friend int compare(vector_mpz const& a, vector_mpz const& b);
		
		/** Returns the rationalization of the given vector. The 
		 *  rationalization is defined to be the vector divided by its first 
		 *  coordinate (if non-zero), or the vector otherwise.
		 */
		vector_mpq rationalization() const;
		
	private:
		/** Internal storage of vector data */
		vector_t v;
		/** Dimension of the vector. */
		size_type d;
	};
	
	int compare(vector_mpz const& a, vector_mpz const& b);
	
	/** Functional to hash a vector_mpz. Compatible with vector_mpq_hash (will 
	 *  hash to the same value for an integer vector_mpq). */
	class vector_mpz_hash 
			: public std::unary_function<vector_mpz, std::size_t> {
	public:
		std::size_t operator() (vector_mpz const& v) const ;
	};
	
	class matrix_mpq_hash;
	
	/** Wraps a multi-precision rational matrix. */
	class matrix_mpq {
	friend class matrix_mpq_hash;
	public:
		/* typedef's for STL compatibility */
		typedef matrix_row_mpq			reference;
		typedef matrix_row_mpq const	const_reference;
		typedef ind						size_type;
		typedef ind						difference_type;
		typedef vector_mpq				value_type;
		typedef vector_mpq*				pointer;
		typedef vector_mpq* const		const_pointer;
		
		class const_iterator;
		/** Row iterator */
		class iterator 
				: public std::iterator<std::random_access_iterator_tag, 
						matrix_mpq::value_type, 
						matrix_mpq::difference_type,
						matrix_mpq::pointer,
						matrix_mpq::reference> {
		friend class matrix_mpq;
		friend class matrix_mpq::const_iterator;
		public:
			
			/** Returns the referenced vector. */
			reference operator* ();
			
			/** Pre-increment operator. */
			iterator& operator++ ();
			/** Post-increment operator. */
			iterator operator++ (int);
			/** Pre-decrement operator. */
			iterator& operator-- ();
			/** Post-decrement operator. */
			iterator operator-- (int);
			/** Addition-assignment operator. */
			iterator& operator+= (difference_type n);
			/** Addition operator. */
			iterator operator+ (difference_type n) const;
			/** Subtraction-assignment operator. */
			iterator& operator-= (difference_type n);
			/** Subtraction operator. */
			iterator operator- (difference_type n) const;
			
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (iterator const& a, 
											  iterator const& b);
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (iterator const& a, 
											  const_iterator const& b);
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (const_iterator const& a, 
											  iterator const& b);
			
			/** Checks if two iterators point to the same row. */
			friend bool operator== (iterator const& a, iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator== (iterator const& a, const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator== (const_iterator const& a, iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (iterator const& a, iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (iterator const& a, const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (const_iterator const& a, iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (iterator const& a, iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (iterator const& a, const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (const_iterator const& a, iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (iterator const& a, iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (iterator const& a, const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (const_iterator const& a, iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (iterator const& a, iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (iterator const& a, const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (const_iterator const& a, iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (iterator const& a, iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (iterator const& a, const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (const_iterator const& a, iterator const& b);
			
		private:
			
			/** Constructor */
			iterator(mpq_class* v, matrix_mpq::size_type d);
			
			/** Pointer to the underlying matrix row */
			mpq_class* v;
			/** Dimension of the matrix rows. */
			matrix_mpq::size_type d;
		};
		
		/** Constant row iterator */
		class const_iterator 
				: public std::iterator<std::random_access_iterator_tag, 
						matrix_mpq::value_type const, 
						matrix_mpq::difference_type,
						matrix_mpq::const_pointer,
						matrix_mpq::const_reference> {
		friend class matrix_mpq;
		friend class matrix_mpq::iterator;
		public:
			
			/** Constifying copy constructor. */
			const_iterator(matrix_mpq::iterator const& that);
			/** Constifying assignment operator. */
			const_iterator& operator= (matrix_mpq::iterator const& that);
			
			/** Returns the referenced vector. */
			reference operator* ();
			
			/** Pre-increment operator. */
			const_iterator& operator++ ();
			/** Post-increment operator. */
			const_iterator operator++ (int);
			/** Pre-decrement operator. */
			const_iterator& operator-- ();
			/** Post-decrement operator. */
			const_iterator operator-- (int);
			/** Addition-assignment operator. */
			const_iterator& operator+= (difference_type n);
			/** Addition operator. */
			const_iterator operator+ (difference_type n) const;
			/** Subtraction-assignment operator. */
			const_iterator& operator-= (difference_type n);
			/** Subtraction operator. */
			const_iterator operator- (difference_type n) const;
			
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (const_iterator const& a, 
											  const_iterator const& b);
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (const_iterator const& a, 
											  matrix_mpq::iterator const& b);
			/** Gets the distance between two iterators. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend difference_type operator- (matrix_mpq::iterator const& a, 
											  const_iterator const& b);
			
			/** Checks if two iterators point to the same row. */
			friend bool operator== (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator== (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator== (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if two iterators point to the same row. */
			friend bool operator!= (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<  (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>  (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator precedes another. Undefined behaviour 
			 *  for iterators not derived from the same matrix. */
			friend bool operator<= (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (const_iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (matrix_mpq::iterator const& a, 
									const_iterator const& b);
			/** Checks if one iterator follows another. Undefined behaviour for 
			 *  iterators not derived from the same matrix. */
			friend bool operator>= (const_iterator const& a, 
									matrix_mpq::iterator const& b);
			
		private:
			
			/** Constructor */
			const_iterator(mpq_class* v, matrix_mpq::size_type d);
			
			/** Pointer to the underlying matrix row */
			mpq_class* v;
			/** Dimension of the matrix rows. */
			matrix_mpq::size_type d;
		};
		
		/** Constructs a matrix with the given number of rows and columns, with 
		 *  all elements initialized to zero. If called as the default 
		 *  constructor, constructs an empty matrix.
		 *  @param n			the number of rows in the matrix (default 0)
		 *  @param d			the number of columns in the matrix (default 0)
		 */
		matrix_mpq(size_type n = 0, size_type d = 0);
		
		/** Copy constructor. 
		 *  @param that			matrix to copy
		 */
		matrix_mpq(matrix_mpq const& that);
		
		/** Destructor */
		~matrix_mpq();
		
		/** Assignment operator.
		 *  @param that			matrix to copy into this one
		 */
		matrix_mpq& operator= (matrix_mpq const& that);
		
		/** Gets an iterator at the first row of the matrix. Note that the 
		 *  vector_mpq returned by this iterator shares storage with this 
		 *  matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made.
		 */
		iterator begin();
		
		/** Gets a constant iterator at the first row of the matrix. Note that 
		 *  the vector_mpq returned by this iterator shares storage with this 
		 *  matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made. */
		const_iterator begin() const;
		
		/** Gets an iterator one past the last row of the matrix. Note that the 
		 *  vector_mpq returned by this iterator shares storage with this 
		 *  matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made. */
		iterator end();
		
		/** Gets a constant iterator one past the last row of the matrix. Note 
		 *  that the vector_mpq returned by this iterator shares storage with 
		 *  this matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made. */
		const_iterator end() const;
		
		/** Gets the number of rows in this matrix. */
		size_type size() const;
		
		/** Gets the dimension of the rows in this matrix. */
		size_type dim() const;
		
		/** Indexing operator. Returns a mutable row reference. Note that the 
		 *  vector_mpq returned by this operator shares storage with this 
		 *  matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made.
		 *  @param i		The index of the row to return.
		 */
		reference operator[] (size_type i);
		
		/** Indexing operator. Returns a constant row reference. Note that the 
		 *  vector_mpq returned by this operator shares storage with this 
		 *  matrix; if a copy of a matrix row is required, it must be 
		 *  explicitly made.
		 *  @param i		The index of the row to return.
		 */
		const_reference operator[] (size_type i) const;
		
		/** Gets the i'th row of this matrix. Synonym for operator[] */
		reference row(size_type i);
		
		/** Gets the i'th row of this matrix. Synonym for operator[] */
		const_reference row(size_type i) const;
		
		/** Gets the (i,j)th element of this matrix */
		mpq_class& elem(size_type i, size_type j);
		
		/** Gets the (i,j)th element of this matrix */
		mpq_class const& elem(size_type i, size_type j) const;
		
		/** Prints the matrix with space-separated rows inside square brackets.
		 *  @param o		The output stream to print on
		 *  @param m		The matrix to print
		 *  @return the same output stream
		 */
		friend std::ostream& operator<< (std::ostream& o, matrix_mpq const& m);
		
		/** Less than. Equivalent to a.compare(b) < 0 */
		friend bool operator<  (matrix_mpq const& a, matrix_mpq const& b);
		/** Equal. Equivalent to a.compare(b) == 0 */
		friend bool operator== (matrix_mpq const& a, matrix_mpq const& b);
		/** Greater than. Equivalent to a.compare(b) > 0 */
		friend bool operator>  (matrix_mpq const& a, matrix_mpq const& b);
		/** Less than or equal. Equivalent to a.compare(b) <= 0 */
		friend bool operator<= (matrix_mpq const& a, matrix_mpq const& b);
		/** Not equal. Equivalent to a.compare(b) != 0 */
		friend bool operator!= (matrix_mpq const& a, matrix_mpq const& b);
		/** Greater than or equal. Equivalent to a.compare(b) >= 0 */
		friend bool operator>= (matrix_mpq const& a, matrix_mpq const& b);
		
		/** Compares two matrices, lexicographically by rows.
		 *  @param a		The first matrix
		 *  @param b		The second matrix
		 *  @return -1 for a less than b, 0 for a equal to b, or 1 for a 
		 *  		greater than b.
		 */
		friend int compare(matrix_mpq const& a, matrix_mpq const& b);
		
		/** Computes the matrix where each entry is the absolute value of the 
		 *  entries of the given matrix.
		 *  @param m		the matrix to take the absolute value of
		 *  @return a matrix R such that R[i][j] = abs(m[i][j])
		 */
		friend matrix_mpq abs(matrix_mpq const& m);
		
		/** Computes the inner product matrix of this matrix.
		 *  @return a matrix P such that P[i][j] = inner_prod(this[i], this[j])
		 */
		matrix_mpq inner_prod_mat();
		
		/** Computes the restriction of the matrix to a given set of row and 
		 *  column indices.
		 *  @param s		The set of indices to restrict the matrix to. The 
		 *  				maximum index in s should be less than or equal to 
		 *  				the smaller of n and d.
		 *  @return a matrix R such that R[i][j] = this[s[i],s[j]]
		 */
		matrix_mpq restriction(index_set s);
	
	private:
		/** Matrix data storage */
		mpq_class* m;
		/** Number of rows in the matrix */
		size_type n;
		/** Dimension of the matrix rows */
		size_type d;
	};
	
	/** Functional to hash a matrix_mpq. Compatible with vector_mpq_hash (will 
	 *  hash to the same value as the concatenation of all the rows of the 
	 *  matrix as a vector_mpq). */
	class matrix_mpq_hash 
			: public std::unary_function<matrix_mpq, std::size_t> {
	public:
		std::size_t operator() (matrix_mpq const& m) const;
	}; /* class matrix_mpq */
	
	int compare(matrix_mpq const& a, matrix_mpq const& b);
	matrix_mpq abs(matrix_mpq const& m);
	
} /* namespace lrs */

#endif /* _LRS_MATRIX_HPP_ */
