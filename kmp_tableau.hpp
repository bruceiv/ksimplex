#pragma once

#include "ksimplex.hpp"

#include "kilomp/kilomp.cuh"

/** 
 * Host-side kilo::mpv based tableau for the KSimplex project.
 * 
 * @author Aaron Moss
 */

namespace ksimplex {

class kmp_tableau {
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
	
	/** Ensures at least a_n limbs are allocated in the matrix */
	void ensure_limbs(u32 a_n) {
		if ( a_n > a_l ) {
			m = kilo::expand(m, m_l, a_l, a_n);
			a_l = a_n;
		}
	}
	
	/** Ensures used limb counter is at least as high as u_n */
	void count_limbs(u32 u_n) {
		if ( u_n > u_l ) { u_l = u_n; }
	}
	
public:	 //public interface
	/**
	 * Default constructor.
	 * 
	 * @param n			The number of equations in the tableau
	 * @param d			The dimension of the underlying space
	 * @param a_l		The number of limbs allocated in the tableau matrix
	 * @param u_l		The maximum number of limbs used of any element in the tableau matrix
	 * @param cob		The indices of the iniital cobasis (should be sorted in increasing order, 
	 * 					cob[0] = 0 (the constant term))
	 * @param bas		The indices of the initial basis (should be sorted in increasing order, 
	 * 					bas[0] = 0 (the objective))
	 * @param mat		The matrix of the initial tableau (should be organized such that the 
	 * 					initial determinant is stored at mat[0], and the variable at row i, 
	 * 					column j is at mat[1+i*d+j], where the 0-row is for the objective function, 
	 * 					and the 0-column is for the constant terms)
	 */
	kmp_tableau(u32 n, u32 d, u32 a_l, u32 u_l, const u32* cob, const u32* bas, kilo::mpv mat)
			: n(n), d(d), a_l(a_l), u_l(u_l), m_l(3 + (n+1)*(d+1)) {
		
		// Allocate basis, cobasis, row, column, and matrix storage
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = kilo::init_mpv(m_l, a_l);
		
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
		kilo::copy(m, mat, 1 + (n+1)*(d+1), u_l);
	}
	
	/**
	 * Copy constructor
	 * 
	 * @param o			The tableau to copy
	 */
	kmp_tableau(const kmp_tableau& o) : n(o.n), d(o.d), a_l(o.a_l), u_l(o.u_l), m_l(o.m_l) {	
		// Allocate basis, cobasis, row, column, and matrix storage
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = kilo::init_mpv(m_l, a_l);
		
		// Copy basis, cobasis, row, column, and matrix
		u32 i;
		for (i = 0; i <= n; ++i) { b[i] = o.b[i]; }
		for (i = 0; i <= d; ++i) { c[i] = o.c[i]; }
		for (i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		kilo::copy(m, o.m, 1 + (n+1)*(d+1), u_l);
	}
	
	/** Destructor */
	~kmp_tableau() {
		delete[] b;
		delete[] c;
		delete[] row;
		delete[] col;
		kilo::clear(m, a_l);
	}
	
	/**
	 * Assignment operator
	 *
	 * @param o			The tableau to assign to this one
	 */
	kmp_tableau& operator = (const kmp_tableau& o) {
		// Ensure matrix storage properly sized
		if ( n == o.n && d == o.d ) {
			// Matrix sizes are compatible, just ensure enough limbs
			u_l = o.u_l;
			ensure_limbs(o.a_l);
		} else {
			// Matrix sizes are not the same, rebuild
			delete[] b;
			delete[] c;
			delete[] row;
			delete[] col;
			kilo::clear(m, a_l);
			
			n = o.n; d = o.d; a_l = o.a_l; u_l = o.u_l; m_l = o.m_l;
			
			b = new u32[n+1];
			c = new u32[d+1];
			row = new u32[n+d+1];
			col = new u32[n+d+1];
			m = kilo::init_mpv(m_l, a_l);
		}
		
		// Copy basis, cobasis, row, column, and matrix
		u32 i;
		for (i = 0; i <= n; ++i) { b[i] = o.b[i]; }
		for (i = 0; i <= d; ++i) { c[i] = o.c[i]; }
		for (i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		kilo::copy(m, o.m, 1 + (n+1)*(d+1), u_l);
		
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
		u32 enter = n+d+1;
		
		u32 i, iL, j, jE;
		
		// Find first cobasic variable with positive objective value
		for (j = 1; j <= d; ++j) {
			if ( kilo::is_pos(m, obj(j)) && c[j] < enter ) {
				enter = c[j];
				jE = j;
			}
		}
		
/*		// Find cobasic variable with maximum positive objective value (break ties by lex-order)
		jE = 0;
		for (j = 1; j <= d; ++j) {
			// Ensure objective value is positive
			if ( kilo::is_pos(m, obj(j)) ) {
				if ( enter == 0 ) {
					// this is the first positive value
					jE = j;
					enter = c[j];
				} else {
					// Check that j has a larger objective value, or the same and a lower index
					s32 t = kilo::cmp(m, obj(j), obj(jE));
					if ( t > 0 || (t == 0 && c[j] < enter) ) {
						jE = j;
						enter = c[j];
					}
				}
			}
		}
*/				
		// If no increasing variables found, this is optimal
		if ( enter == n+d+1 ) return tableau_optimal;
		
		u32 iMin = 0;
		u32 leave = 0;
		u32 t1 = tmp(1);
		u32 t2 = tmp(2);
		
		ensure_limbs(u_l*2);  // Make sure enough space in temp variables
		
		for (iL = d+1; iL <= n; ++iL) {  // Ignore decision variables (first d)
			if ( kilo::is_neg(m, elm(iL, jE)) ) {  // Negative value in entering column
				if ( leave == 0 ) {  // First possible leaving variable
					iMin = iL;
					leave = b[iL];
				} else {  // Test against previous leaving variable
					i = b[iL];
					
					//compute ratio: rat = M[iMin, 0] * M[iL, jE] <=> M[iL, 0] * M[iMin, jE]
					kilo::mul(m, t1, con(iMin), elm(iL, jE));
					kilo::mul(m, t2, con(iL), elm(iMin, jE));
					s32 rat = kilo::cmp(m, t1, t2);
					
					//test ratio
					if ( rat == -1 || ( rat == 0 && i < leave ) ) {
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
		
		ensure_limbs(u_l*2);       // Make sure enough space in temp variables
		
		// Keep sign of M[iL,jE] in det
		u32 Mij = elm(iL, jE);
		if ( kilo::is_neg(m, Mij) ) { kilo::neg(m, det); }
		
		// Recalculate matrix elements outside of pivot row/column
		for (i = 0; i <= n; ++i) {
			if ( i == iL ) continue;
			
			u32 Mi = elm(i, jE);
			for (j = 0; j <= d; ++j) {
				if ( j == jE ) continue;
				
				u32 Eij = elm(i, j);
				
				// M[i,j] = ( M[i,j]*M[iL,jE] - M[i,jE]*M[iL,j] )/det
				kilo::mul(m, t1, Eij, Mij);
				kilo::mul(m, Eij, Mi, elm(iL, j));
				kilo::sub(m, t1, Eij);
				count_limbs(kilo::div(m, Eij, t1, det));  //store # of limbs
			}
		}
		
		// Recalculate pivot row/column
		if ( kilo::is_pos(m, Mij) ) {
			for (j = 0; j <= d; ++j) {
				kilo::neg(m, elm(iL, j));
			}
		} else { // M[iL,jE] < 0 -- == 0 case is ruled out by pre-assumptions
			for (i = 0; i <= n; ++i) {
				kilo::neg(m, elm(i, jE));
			}
		}
		
		// Reset pivot element, determinant
		kilo::swap(m, det, Mij);
		if ( kilo::is_neg(m, det) ) { kilo::neg(m, det); }
		
		// Fix basis, cobasis, row, and column
		b[iL] = enter;
		c[jE] = leave;
		row[leave] = 0;
		row[enter] = iL;
		col[enter] = 0;
		col[leave] = jE;
	}

	/** Get a read-only matrix copy */
	const kilo::mpv mat() const { return m; }
	
private:  //class members
	u32 n;        ///< number of equations in tableau
	u32 d;        ///< dimension of underlying space
	
	u32* b;       ///< basis variables
	u32* c;       ///< cobasis variables
	
	u32* row;     ///< row indices for variables
	u32* col;     ///< column indices for variables
	
	kilo::mpv m;  ///< underlying matrix for tableau
	u32 a_l;      ///< number of limbs allocated for matrix
	u32 u_l;      ///< maximum number of limbs used for matrix
	u32 m_l;      ///< number of elements in the matrix
}; /* class kmp_tableau */

} /* namespace ksimplex */
