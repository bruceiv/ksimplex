#pragma once

/**
 * LRS-based tableau for verification of the KSimplex project.
 *
 * @author Aaron Moss
 */

#include <utility>

#include "ksimplex.hpp"

#include "lrs/clrs.hpp"
#include "lrs/lrs.hpp"
#include "lrs/matrix.hpp"

namespace ksimplex {

class lrs_tableau {
public:
	/**
	 * Default constructor.
	 * @param l			LRS wrapper to use - should be initialized and moved to first basis
	 */
	lrs_tableau(lrs::lrs& l, u32 n, u32 d) : l(l), n(n), d(d) {}
	
	pivot ratioTest() {
		std::pair<u64,u64> p = l.blandRatio();
std::cout << "\tlrs::blandRatio() -> (" << p.first << "," << p.second << ")" << std::endl;
		pivot q(p.first, p.second);
		return q;
	}
	
	void doPivot(u32 enter, u32 leave) {
		l.pivot(leave, enter);
	}
	
	/** @return a lrs::vector_mpz in the same layout as the matrix returned from kmp_tableau */
	const lrs::vector_mpz mat() const {
		lrs::vector_mpz v(1+(n+1)*(d+1));
		
		// Copy determinant
		lrs::copy(v[0], l.getDeterminant());
		u32 k = 1;
		for (u32 i = 0; i <= n; ++i) {
			// Copy constant term
			lrs::copy(v[k++], l.elem(i, d));
			
			// Copy matrix elements
			for (u32 j = 0; j < d; ++j) { lrs::copy(v[k++], l.elem(i, j)); }
		}
		
		return v;
	}
private:
	lrs::lrs& l;  ///< The LRS wrapper object
	u32 n;        ///< number of equations in tableau
	u32 d;        ///< dimension of underlying space
}; /* class lrs_tableau */

} /* namespace ksimplex */

