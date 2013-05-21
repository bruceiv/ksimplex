#pragma once

#include <gmp.h>

#include "gmpv.hpp"
#include "ksimplex.hpp"

/** 
 * GMP based tableau for the KSimplex project.
 * 
 * @author Aaron Moss
 */

namespace ksimplex {

class gmp_tableau {
private:  //internal convenience functions
	
	/** @return index of the determinant */
	static const u32 det = 0;
	
	/** @return index of j'th objective function coefficient (j >= 1) */
	inline u32 obj(u32 j) const { return j+1; }
	
	/** @return index of constant coefficient of i'th row (i >= 1) */
	inline u32 con(u32 i) const { return i*(d+1)+1; }
	
	/** @return index of j'th coefficient of i'th row (i, j >= 1) */
	inline u32 elm(u32 i, u32 j) const { return i*(d+1)+j+1; }
	
	/** @return index of x'th temp variable (x >= 1) */
	inline u32 tmp(u32 x) const { return (n+1)*(d+1)+x; }
	
public:	 //public interface
	/**
	 * Default constructor.
	 * 
	 * @param n			The number of equations in the tableau
	 * @param d			The dimension of the underlying space
	 * @param cob		The indices of the iniital cobasis (should be sorted in increasing order, 
	 * 					cob[0] = 0 (the constant term))
	 * @param bas		The indices of the initial basis (should be sorted in increasing order, 
	 * 					bas[0] = 0 (the objective))
	 * @param mat		The matrix of the initial tableau (should be organized such that the 
	 * 					initial determinant is stored at mat[0], and the variable at row i, 
	 * 					column j is at mat[1+i*d+j], where the 0-row is for the objective function, 
	 * 					and the 0-column is for the constant terms)
	 */
	gmp_tableau(u32 n, u32 d, const u32* cob, const u32* bas, mpz_t* mat)
			: n(n), d(d), m_l(3 + (n+1)*(d+1)) {
		
		// Allocate basis, cobasis, row, column, and matrix storage
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = init_gmpv(m_l);
		
		u32 i, j, r_i, c_j;
		
		// Copy basis and row indices
		b[0] = 0;
		r_i = 0;
		for (i = 1; i <= n; ++i) {
			b[i] = bas[i];
			while ( r_i < bas[i] ) row[r_i++] = 0;
			row[r_i++] = i;
		}
		while ( r_i <= n+d ) row[r_i++] = 0;
		
		// Copy cobasis and column indices
		c[0] = 0;
		c_j = 0;
		for (j = 1; j <= d; ++j) {
			c[j] = cob[j];
			while ( c_j < cob[j] ) col[c_j++] = 0;
			col[c_j++] = j;
		}
		while ( c_j <= n+d ) col[c_j++] = 0;
		
		// Copy matrix
		copy_gmpv(m, mat, 1 + (n+1)*(d+1));
	}
	
	/**
	 * Copy constructor
	 * 
	 * @param o			The tableau to copy
	 */
	gmp_tableau(const gmp_tableau& o) : n(o.n), d(o.d), m_l(o.m_l) {	
		// Allocate basis, cobasis, row, column, and matrix storage
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = init_gmpv(m_l);
		
		// Copy basis, cobasis, row, column, and matrix
		u32 i;
		for (i = 0; i <= n; ++i) { b[i] = o.b[i]; }
		for (i = 0; i <= d; ++i) { c[i] = o.c[i]; }
		for (i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		copy_gmpv(m, o.m, 1 + (n+1)*(d+1));
	}
	
	/** Destructor */
	~gmp_tableau() {
		delete[] b;
		delete[] c;
		delete[] row;
		delete[] col;
		clear_gmpv(m, m_l);
	}
	
	/**
	 * Assignment operator
	 *
	 * @param o			The tableau to assign to this one
	 */
	gmp_tableau& operator = (const gmp_tableau& o) {
		// Ensure matrix storage properly sized
		if ( n != o.n || d != o.d ) {
			// Matrix sizes are not the same, rebuild
			delete[] b;
			delete[] c;
			delete[] row;
			delete[] col;
			clear_gmpv(m, m_l);
			
			n = o.n; d = o.d; m_l = o.m_l;
			
			b = new u32[n+1];
			c = new u32[d+1];
			row = new u32[n+d+1];
			col = new u32[n+d+1];
			m = init_gmpv(m_l);
		}
		
		// Copy basis, cobasis, row, column, and matrix
		u32 i;
		for (i = 0; i <= n; ++i) { b[i] = o.b[i]; }
		for (i = 0; i <= d; ++i) { c[i] = o.c[i]; }
		for (i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		copy_gmpv(m, o.m, 1 + (n+1)*(d+1));
		
		return *this;
	}
	
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
			if ( jE != 0 && mpz_sgn(m[obj(jE)]) > 0 ) {
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
		
		for (iL = 1; iL <= n; ++iL) {
			if ( mpz_sgn(m[elm(iL, jE)]) < 0 ) {  // Negative value in entering column
				if ( leave == 0 ) {  // First possible leaving variable
					iMin = iL;
					leave = b[iL];
				} else {  // Test against previous leaving variable
					i = b[iL];
					
					//compute ratio: rat = M[iMin, 0] * M[iL, jE] <=> M[iL, 0] * M[iMin, jE]
					mpz_mul(m[t1], m[con(iMin)], m[elm(iL, jE)]);
					mpz_mul(m[t2], m[con(iL)], m[elm(iMin, jE)]);
					int rat = mpz_cmp(m[t1], m[t2]);
					
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
	
	/** 
	 * Pivots the tableau from one basis to another.
	 * The caller is responsible to ensure that this is a valid pivot (i.e. the given entering 
	 * variable is cobasic, leaving variable is basic, and coefficient of the entering variable in 
	 * the equation defining the leaving variable is non-zero).
	 * 
	 * @param enter		The index to enter the basis
	 * @param leave		The index to leave the basis
	 */
	void doPivot(u32 enter, u32 leave) {
		u32 iL = row[leave];  // Leaving row
		u32 jE = col[enter];  // Entering column
		
		u32 i, j;
		u32 t1 = tmp(1);
		
		// Keep sign of M[iL,jE] in det
		u32 Mij = elm(iL, jE);
		if ( mpz_sgn(m[Mij]) < 0 ) { mpz_neg(m[det], m[det]); }
		
		// Recalculate matrix elements outside of pivot row/column
		for (i = 0; i <= n; ++i) {
			if ( i == iL ) continue;
			
			u32 Mi = elm(i, jE);
			for (j = 0; j <= d; ++j) {
				if ( j == jE ) continue;
				
				u32 Eij = elm(i, j);
				
				// M[i,j] = ( M[i,j]*M[iL,jE] - M[i,jE]*M[iL,j] )/det
				mpz_mul(m[t1], m[Eij], m[Mij]);
				mpz_mul(m[Eij], m[Mi], m[elm(iL, j)]);
				mpz_sub(m[t1], m[t1], m[Eij]);
				mpz_divexact(m[Eij], m[t1], m[det]);
			}
		}
		
		// Recalculate pivot row/column
		if ( mpz_sgn(m[Mij]) > 0 ) {
			for (j = 0; j <= d; ++j) {
				mpz_neg(m[elm(iL, j)], m[elm(iL, j)]);
			}
		} else { // M[iL,jE] < 0 -- == 0 case is ruled out by pre-assumptions
			for (i = 0; i <= n; ++i) {
				mpz_neg(m[elm(i, jE)], m[elm(i, jE)]);
			}
		}
		
		// Reset pivot element, determinant
		mpz_swap(m[det], m[Mij]);
		if ( mpz_sgn(m[det]) < 0 ) { mpz_neg(m[det], m[det]); }
		
		// Fix basis, cobasis, row, and column
		b[iL] = enter;
		c[jE] = leave;
		row[leave] = 0;
		row[enter] = iL;
		col[enter] = 0;
		col[leave] = jE;
	}

	/** Get a read-only matrix copy */
	const mpz_t* mat() const { return m; }
	
private:  //class members
	u32 n;        ///< number of equations in tableau
	u32 d;        ///< dimension of underlying space
	
	u32* b;       ///< basis variables
	u32* c;       ///< cobasis variables
	
	u32* row;     ///< row indices for variables
	u32* col;     ///< column indices for variables
	
	mpz_t* m;     ///< underlying matrix for tableau
	u32 m_l;      ///< number of elements in the matrix
}; /* class gmp_tableau */

} /* namespace ksimplex */
