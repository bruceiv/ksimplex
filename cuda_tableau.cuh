#pragma once

#include "ksimplex.hpp"

#include "kilomp/kilomp.cuh"
#include "kilomp/kilomp_cuda.cuh"

/** 
 * Device-side kilo::mpv based tableau for the KSimplex project.
 * 
 * @author Aaron Moss
 */

namespace ksimplex {

class cuda_tableau {
private:  //internal convenience functions
	
	/** Ensures at least a_n limbs are allocated in the device matrix */
	void ensure_limbs_d(u32 a_n) {
		if ( a_n > a_dl ) {
			m = kilo::expand_d(m, m_dl, a_dl, a_n);
			a_dl = a_n;
		}
	}
	
	/** Ensures that there is enough space in the matrix to hold temporaries of all current 
	 *  calculations. */
	void ensure_temp_space_d() {
		cudaMemcpy(&u_l, u_d, sizeof(u32), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		ensure_limbs_d(2*u_l);
	}
	
	/** Ensures at least a_n limbs are allocated in the host matrix */
	void ensure_limbs(u32 a_n) {
		if ( a_n > a_hl ) {
			m = kilo::expand(m, m_hl, a_hl, a_n);
			a_hl = a_n;
		}
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
	cuda_tableau(u32 n, u32 d, u32 a_l, u32 u_l, const u32* cob, const u32* bas, kilo::mpv mat)
			: n(n), d(d), a_hl(a_l), a_dl(a_l), u_l(u_l), 
			m_dl(1 + 2*(n+1)*(d+1)), m_hl(1+ (n+1)*(d+1)) {
		
		// Allocate basis, cobasis, row, column, and matrix storage on host
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = kilo::init_mpv(m_hl, a_hl);
		
		// Allocate limb count, basis, cobasis, and matrix storage on device
		u_d = cudaMalloc((void**)&u_d, sizeof(u32)); CHECK_CUDA_SAFE
		b_d = cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
		c_d = cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
		m_d = kilo::init_mpv_d(m_dl, a_dl);
		
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
		
		// Copy limb count, basis, cobasis, and matrix to device
		cudaMemcpy(u_d, &u_l, sizeof(u32), cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(b_d, b, (n+1)*sizeof(u32), cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(c_d, c, (d+1)*sizeof(u32), cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		kilo::copy_hd(m_d, mat, m_hl, u_l);
	}
	
	/**
	 * Copy constructor
	 * 
	 * @param o			The tableau to copy
	 */
	cuda_tableau(const cuda_tableau& o)
			: n(o.n), d(o.d), a_hl(o.a_dl), a_dl(o.a_dl), u_l(o.u_l), m_dl(o.m_dl), m_hl(o.m_hl) {	
		// Allocate basis, cobasis, row, column, and matrix storage on host
		b = new u32[n+1];
		c = new u32[d+1];
		row = new u32[n+d+1];
		col = new u32[n+d+1];
		m = kilo::init_mpv(m_hl, a_hl);
		
		// Allocate limb count, basis, cobasis, and matrix storage on device
		u_d = cudaMalloc((void**)&u_d, sizeof(u32)); CHECK_CUDA_SAFE
		b_d = cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
		c_d = cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
		m_d = kilo::init_mpv_d(m_dl, a_dl);
		
		// Copy row and column on host
		for (u32 i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (u32 i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		
		// Copy limb count, basis, cobasis, and matrix on device
		cudaMemcpy(u_d, o.u_d, sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(b_d, o.b_d, (n+1)*sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(c_d, o.c_d, (d+1)*sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		kilo::copy_dd(m_d, o.m_d, m_hl, u_l);
	}
	
	/** Destructor */
	~cuda_tableau() {
		// Clear host-side storage
		delete[] b;
		delete[] c;
		delete[] row;
		delete[] col;
		kilo::clear(m, a_hl);
		
		// Clear device-side storage
		cudaFree(u_d); CHECK_CUDA_SAFE
		cudaFree(b_d); CHECK_CUDA_SAFE
		cudaFree(c_d); CHECK_CUDA_SAFE
		kilo::clear_d(m_d, a_dl);
	}
	
	/**
	 * Assignment operator
	 *
	 * @param o			The tableau to assign to this one
	 */
	cuda_tableau& operator = (const cuda_tableau& o) {
		// Ensure matrix storage properly sized
		if ( n == o.n && d == o.d ) {
			// Matrix sizes are compatible, just ensure enough limbs on device
			u_l = o.u_l;
			ensure_limbs_d(o.a_dl);
		} else {
			// Matrix sizes are not the same, rebuild
			// Clear host-side storage
			delete[] b;
			delete[] c;
			delete[] row;
			delete[] col;
			kilo::clear(m, a_hl);
		
			// Clear device-side storage
			cudaFree(b_d); CHECK_CUDA_SAFE
			cudaFree(c_d); CHECK_CUDA_SAFE
			kilo::clear_d(m_d, a_dl);
			
			n = o.n; d = o.d; a_hl = o.a_dl; a_dl = o.a_dl; u_l = o.u_l; 
			m_hl = o.m_hl; m_dl = o.m_dl;
			
			// Allocate basis, cobasis, row, column, and matrix storage on host
			b = new u32[n+1];
			c = new u32[d+1];
			row = new u32[n+d+1];
			col = new u32[n+d+1];
			m = kilo::init_mpv(m_hl, a_hl);
		
			// Allocate basis, cobasis, and matrix storage on device
			b_d = cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
			c_d = cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
			m_d = kilo::init_mpv_d(m_dl, a_dl);
		}
		
		// Copy row and column on host
		for (u32 i = 0; i <= n+d; ++i) { row[i] = o.row[i]; }
		for (u32 i = 0; i <= n+d; ++i) { col[i] = o.col[i]; }
		
		// Copy limb count, basis, cobasis, and matrix on device
		cudaMemcpy(u_d, o.u_d, sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(b_d, o.b_d, (n+1)*sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(c_d, o.c_d, (d+1)*sizeof(u32), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		kilo::copy_dd(m_d, o.m_d, m_hl, u_l);
		
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
	u32 n;                ///< number of equations in tableau
	u32 d;                ///< dimension of underlying space
	
	u32* b_d;             ///< basis on device
	u32* c_d;             ///< cobasis on device
	kilo::mpv m_d;        ///< tableau matrix on device
	
	mutable u32* b;       ///< host-side basis variable buffer
	mutable u32* c;       ///< host-side cobasis variable buffer
	mutable kilo::mpv m;  ///< host-side matrix buffer
	
	u32* row;             ///< row indices for variables
	u32* col;             ///< column indices for variables
	
	u32 a_hl;             ///< number of limbs allocated for host matrix
	u32 a_dl;             ///< number of limbs allocated for device matrix
	u32 u_l;              ///< maximum number of limbs used for matrix
	u32* u_d;             ///< device storage for number of used limbs
	u32 m_dl;             ///< number of elements in the device matrix (includes temps)
	u32 m_hl;             ///< number of elements in the host matrix (excludes temps)
	
}; /* class cuda_tableau */

} /* namespace ksimplex */
