#ifndef _COBASIS_H_
#define _COBASIS_H_

#include <cstdlib>
#include <functional>
#include <iterator>
#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <gmpxx.h>

#include "clrs.hpp"
#include <string.h>

namespace lrs {
	
	/** Represents a basis or cobasis of the dictionary. For an index set s, 
	 *  s[i] == true iff index i is included in the basis or cobasis.
	 */
	typedef boost::dynamic_bitset<> index_set;
	
	
	/** Iterator to represent an index set as a sequence of indices, rather 
	 *  than a set of boolean values.
	 */
	class index_set_iter 
			: public boost::iterator_facade<
					index_set_iter, ind, std::input_iterator_tag, ind> {
	public:
		/* Uses default copy, etc. constructors */
		
		/** Initialize this iterator on the given set with a starting index of 
		 *  the first one bit in the set. Note that this does not preclude the 
		 *  possibility that there may be no one bits, in which case the 
		 *  iterator will point to the end of the series.
		 */
		index_set_iter(index_set const* s) : s(s), i(s->find_first()) { }
		
		/** Initialize this iterator on the given set with a starting index
		 *  no less than the given index. 
		 *  @param s		The set to initialize the iterator on
		 *  @param ind		The index to start the iterator on. If this index 
		 * 					does not mark a value, the next such value will be 
		 * 					chosen.
		 */
		index_set_iter(index_set const* s, uind i) : s(s), i(i) { 
			uind u_i = i;
			if ( u_i < s->size() && ! s->test(u_i) ) i = s->find_next(u_i);
		}
		
	private:
		/* required for boost::iterator_facade */
		friend class boost::iterator_core_access;
		
		/** Sets the index to the next set bit. In compliance with 
		 *  boost::iterator_facade
		 */
		void increment()
			{ i = s->find_next(i); }
		
		/** Tests the equality of two iterators. In compliance with 
		 *  boost::iterator_facade
		 */
		bool equal(index_set_iter const& o) const 
			{ return s == o.s && i == o.i; }
		
		/** Returns the index of the current set bit, to allow the bitset to 
		 *  simulate a list of indices. In compliance with 
		 *  boost::iterator_facade.
		 */
		ind dereference() const 
			{ return ind(i); }
		
		/** set this iterator is defined on */
		index_set const* s;
		/** index of last seen value */
		uind i;
	};
	
	/** Functional to hash an index_set */
	class index_set_hash : public std::unary_function<index_set, std::size_t> {
	public:
		/** Hash function for an index set. XOR's the blocks of the set 
		 *  together, and returns the result.
		 *  @param s		The set to hash
		 *  @return the hash value
		 */
		std::size_t operator() (index_set const& s) const {
			/* set up the XOR functional */
			xor_fun x;
			/* read the block range into the XOR iterator */
			boost::to_block_range(s, boost::make_function_output_iterator(x));
			/* return the final value */
			return x.val;
		}
	private:
		/** Functor that XOR's its argument with its internal state. */
		class xor_fun {
		public:
			xor_fun() : val(0UL) {}
			
			void operator() (index_set::block_type const& x) { val ^= x; }
			
			std::size_t val;
		};
	};
	
	/** Gets the first iterator for the index set. */
	static index_set_iter begin(index_set const& s) 
		{ return index_set_iter(&s); }
	
	/** Gets the end iterator for the index set. */
	static index_set_iter end(index_set const& s) 
		{ return index_set_iter(&s, s.npos); }
	
	/** Gets a pseudo-random index from an index set. For an index set s, 
	 *  s.test(pseudoRandomInd(s)) is guaranteed to return true, but no 
	 *  guarantee is placed on the randomness of the distribution. Performance 
	 *  of the algorithm will be O(size(s)), but more performant algorithms 
	 *  will be preferred to more mathematically random algorithms.
	 */
	static ind pseudoRandomInd(index_set& s) {
		/* The first index set */
		uind firstInd = s.find_first();
		/* The number of set indices that are possibly set */
		uind setSize = s.size() - firstInd;
		
		/* generate a random index in the range [ firstInd , s.size() ) */
		uind randInd = firstInd + ( rand() * ( setSize - 1 ) / RAND_MAX );
		if ( ! s.test(randInd) ) {
			/* if this is not an element of the set, find the next element */
			randInd = s.find_next(randInd);
			/* if this overflowed the set, use first element */
			if ( randInd >= s.size() ) randInd = firstInd;
		}
		/* note that this will be biased toward indices with large empty ranges 
		 * preceding them, and is still worst-case linear, but should be 
		 * acceptably distributed. */
		return ind(randInd);
	}
	
	/** Lexicographically compares two index sets. This comparison views the 
	 *  sets as sets of indices, rather than boost::dynamic_bitset's usual 
	 *  comparison, which acts on them as integers)
	 *  @return a < b */
	static bool lexicographical_compare(index_set const& a, 
										index_set const& b) {
		ind t;
		
		for (index_set_iter iterA = lrs::begin(a), iterB = lrs::begin(b); 
				iterA != lrs::end(a), iterB != lrs::end(b); ++iterA, ++iterB) {
			t = *iterA - *iterB;
			if (t < 0) return true; else if (t > 0) return false;
		}
		
		// if it reaches here, the two are lexicographically equal up to the 
		// end of the shorter vector
		
		return a.count() < b.count();
	}
	
	
	/** Stores a cobasis and related information.
	 */
	struct cobasis {
		
		cobasis(mpz_class& det, ind ray, index_set& cob, ind totalInc, 
				index_set& extraInc) : det(det), ray(ray), cob(cob), 
				totalInc(totalInc), extraInc(extraInc) {}
		
		/** matrix determinant */
		mpz_class det;
		/** ray index */
		ind ray;
		/** cobasis */
		index_set cob;
		/** total number of incident facets */
		ind totalInc;
		/** extra incident facet indexes */
		index_set extraInc;
	};
	
} /* namespace lrs */

#endif /* _COBASIS_H_ */
