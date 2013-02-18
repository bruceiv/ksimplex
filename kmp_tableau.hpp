#pragma once

#include "ksimplex.hpp"

/** 
 * Host-side kilo::mpv based tableau for the KSimplex project.
 * 
 * @author Aaron Moss
 */

namespace ksimplex {

class kmp_tableau {
private:  //internal convenience functions
	/** @return index of j'th objective function coefficient (j >= 1) */
	u32 obj(u32 j) { return j+1; }
	
	/** @return index of constant coefficient of i'th row (i >= 1) */
	u32 con(u32 i) { return i*(d+1)+1; }
	
	/** @return index of j'th coefficient of i'th row (i, j >= 1) */
	u32 elm(u32 i, u32 j) { return i*(d+1)+j+1; }
	
	/** @return index of x'th temp variable (x >= 1) */
	u32 tmp(u32 x) { return (n+1)*(d+1)+x; }
	
	/** Ensures at least a_n limbs are allocated in the matrix */
	void ensure_limbs(u32 a_n) {
		if ( a_n > a_l ) {
			m = kilo::expand(m, m_l, a_l, a_n);
			a_l = a_n;
		}
	}
public:	 //public interface
	
	/** 
	 * Finds the next pivot using Bland's rule.
	 * 
	 * @return The next pivot by Bland's rule, tableau_optimal if no such pivot because the tableau 
	 *         is optimal, or tableau_unbounded if no such pivot because the tableau is unbounded.
	 */
	pivot ratioTest() {
		// Look for entering variable
		u32 enter = 0;
		
		u32 i, iL, j, jE;
		
		// Find first cobasic variable with positive objective value
		for (j = 1; j <= n+d; ++j) {
			jE = col[j];  // Get column index of variable j
			
			// Check that objective value for j is positive
			if ( jE != 0 && kilo::is_pos(m, obj(jE)) ) {
				enter = j;
				break;
			}
		}
		
		// If no increasing variables found, this is optimal
		if ( enter == 0 ) return tableau_optimal;
		
		u32 iMin = 0;
		u32 leave = 0;
		u32 t1 = tmp(1);
		u32 t2 = tmp(2);
		
		ensure_limbs(u_l*2);  // Make sure enough space in temp variables
		
		for (iL = 1; iL <= n; ++iL) {
			if ( kilo::is_neg(m, elm(iL, jE)) ) {  // Negative value in entering column
				if ( leave == 0 ) {  // First possible leaving variable
					iMin = iL;
					leave = b[iL];
				} else {  // Test against previous leaving variable
					i = b[iL];
					
					//compute ratio
					kilo::mul(m, t1, con(iMin), elm(iL, jE));
					kilo::mul(m, t2, con(iL), elm(iMin, jE));
					s32 rat = kilo::cmp(m, t1, t2);
					
					//test ratio
					if ( rat < 0 || ( rat == 0 && i < leave ) ) {
						iMin = iL;
						leave = i;
					}
				}
			}
		}
		
		// If no limiting variables found, this is unbounded
		if ( leave == 0 ) return tableau_unbounded;
		
		// Return pivot
		return pivot(enter, leave);
	}
	
private:  //class members
	u32 n;     ///< number of equations in tableau
	u32 d;     ///< dimension of underlying space
	
	u32* b;    ///< basis variables
	u32* c;    ///< cobasis variables
	
	u32* row;  ///< row indices for variables
	u32* col;  ///< column indices for variables
	
	mpv m;     ///< underlying matrix for tableau
	u32 a_l;   ///< number of limbs allocated for matrix
	u32 u_l;   ///< maximum number of limbs used for matrix
	u32 m_l;   ///< number of elements in the matrix
}; /* class kmp_tableau */

} /* namespace ksimplex */
