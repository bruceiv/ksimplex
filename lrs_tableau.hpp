#pragma once

/**
 * LRS-based tableau for verification of the KSimplex project.
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
		pivot q(p.second, p.first);
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
	
	/** @return the objective value of this tableau */
	const lrs::val_t& obj() const { return l.elem(0, d); }
	
	/** @return the determinant of this tableau */
	const lrs::val_t& det() const { return l.getDeterminant(); }
private:
	lrs::lrs& l;  ///< The LRS wrapper object
	u32 n;        ///< number of equations in tableau
	u32 d;        ///< dimension of underlying space
}; /* class lrs_tableau */

} /* namespace ksimplex */

