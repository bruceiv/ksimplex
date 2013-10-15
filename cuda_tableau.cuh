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

/** @return index of j'th objective function coefficient (j >= 1) */
__device__ inline u32 obj(u32 j, u32 n, u32 d) { return j+1; }
/** @return index of constant coefficient of i'th row (i >= 1) */
__device__ inline u32 con(u32 i, u32 n, u32 d) { return i*(d+1)+1; }
/** @return index of the j'th coefficient of the i'th row (i, j >= 1) */
__device__ inline u32 elm(u32 i, u32 j, u32 n, u32 d) { return i*(d+1)+j+1; }
/** @return index of x'th temp variable (x >= 1) */
__device__ inline u32 tmp(u32 x, u32 n, u32 d) { return (n+1)*(d+1)+x; }

/** 
 * Finds the smallest index of a cobasic variable having a positive objective value. 
 * Intended to be private to cuda_tableau class; should be called with one block of 2^k threads, 
 * for some k.
 *  
 * @param blockSize  Block size of the invocation
 * @param m_d        The matrix on device
 * @param c_d        The cobasis index list on device (should index up to at least d)
 * @param o_d        Output parameter - will hold the cobasic index (0 for nonesuch)
 * @param n          The maximum valid row index
 * @param d          The maximum valid column index
 */
template<u32 blockSize> 
__global__ void posObj_k(kilo::mpv m_d, u32* c_d, u32* o_d, u32 n, u32 d) {
	
	// Column index for first entering variable which improves objective
	__shared__ u32 enter[blockSize];
	
	// Get column index
	u32 tid = threadIdx.x;
	
	// Initialize reduction to value one higher than maximum
	enter[tid] = n+d+1;
	
	// Check value at this thread's columns
	if ( tid < d && kilo::is_pos(m_d, obj(tid+1, n, d)) ) {
		enter[tid] = c_d[tid+1];
	}
	for (u32 j = tid+blockSize+1; j <= d; j += blockSize) {
		if ( c_d[j] < enter[tid] && kilo::is_pos(m_d, obj(j, n, d)) ) {
			enter[tid] = c_d[j];
		}
	}
	__syncthreads();
	
	// Reduce (with last warp unrolled)
	for (u32 s = blockSize >> 1; s > 32; s >>= 1) {
		if ( tid < s ) {
			if ( enter[tid] > enter[tid+s] ) enter[tid] = enter[tid+s];
		}
		
		__syncthreads();
	}
	if ( blockSize >= 64 && tid < 32 && enter[tid] > enter[tid+32] ) enter[tid] = enter[tid+32];
	if ( blockSize >= 32 && tid < 16 && enter[tid] > enter[tid+16] ) enter[tid] = enter[tid+16];
	if ( blockSize >= 16 && tid <  8 && enter[tid] > enter[tid+ 8] ) enter[tid] = enter[tid+ 8];
	if ( blockSize >=  8 && tid <  4 && enter[tid] > enter[tid+ 4] ) enter[tid] = enter[tid+ 4];
	if ( blockSize >=  4 && tid <  2 && enter[tid] > enter[tid+ 2] ) enter[tid] = enter[tid+ 2];
	// Combine last step of reduction with output
	if ( tid == 0 ) {
		if ( enter[0] > enter[1] ) *o_d = enter[1];
		else *o_d = enter[0];
	}
}

/**
 * Finds the leaving variable which most improves the objective for a given entering variable. 
 * Intended to be private to cuda_tableau class; should be called with one block of 2^k threads, 
 * for some k.
 *  
 * @param blockSize  Block size of the invocation
 * @param m_d        The matrix on device
 * @param u_d        The device-side limb counter
 * @param jE         Column index of entering variable
 * @param b_d        Basis buffer on device
 * @param o_d        Output parameter - will hold the basic index (0 for nonesuch)
 * @param n          The maximum valid row index
 * @param d          The maximum valid column index
 */
template<u32 blockSize>
__global__ void minRatio_k(kilo::mpv m_d, u32* u_d, u32 jE, u32* b_d, u32* o_d, u32 n, u32 d) {
	// Row index for leaving variable which improves objective by maximum 
	// amount
	__shared__ u32 leave[blockSize];
	
	u32 tid = threadIdx.x;
	
	// First find min ratios for each thread
	leave[tid] = 0;
	u32 t1 = tmp(tid+1, n, d);
	u32 t2 = tmp(tid+blockSize+1, n, d);
	
	for (u32 iL = tid+d+1; iL <= n; iL += blockSize) {  // Ignore decision variables (first d)
		if ( kilo::is_neg(m_d, elm(iL, jE, n, d)) ) {  // Negative value in entering column
			if ( leave[tid] == 0 ) {  // First possible leaving variable
				leave[tid] = iL;
			} else {  // Test against previous leaving variable
				u32 iMin = leave[tid];
				
				// Compute ratio: rat = M[iMin, 0] * M[iL, jE] <=> M[iL, 0] * M[iMin, jE]
				kilo::mul(m_d, t1, con(iMin, n, d), elm(iL, jE, n, d));
				kilo::mul(m_d, t2, con(iL, n, d), elm(iMin, jE, n, d));
				s32 rat = kilo::cmp(m_d, t1, t2);
				
				// Test ratio
				if ( rat == -1 || ( rat == 0 && b_d[iL] < b_d[iMin] ) ) {
					leave[tid] = iL;
				}
			}
		}
	}
	__syncthreads();
	
	// Reduce
	for (u32 s = blockSize >> 1; s > 0; s >>= 1) {
		if ( tid < s ) {
			if ( leave[tid+s] != 0 ) {
				if ( leave[tid] == 0 ) {
					leave[tid] = leave[tid+s];
				} else {
					u32 iMin = leave[tid], iL = leave[tid+s];
					
					// Compute ratio: rat = M[iMin, 0] * M[iL, jE] <=> M[iL, 0] * M[iMin, jE]
					kilo::mul(m_d, t1, con(iMin, n, d), elm(iL, jE, n, d));
					kilo::mul(m_d, t2, con(iL, n, d), elm(iMin, jE, n, d));
					s32 rat = kilo::cmp(m_d, t1, t2);
				
					// Test ratio
					if ( rat == -1 || ( rat == 0 && b_d[iL] < b_d[iMin] ) ) {
						leave[tid] = iL;
					}
				}
			}
		}
		
		if ( s > 32 ) __syncthreads(); // syncronize up to last warp
	}
	
	// Report minimum ratio
	if ( tid == 0 ) {
		*o_d = b_d[leave[0]];
	}
}

/**
 * Pivots the tableau on device. Substitues all values but those in the leaving row and entering 
 * column. Intended to be private to cuda_tableau class; should be called with n blocks of d 
 * threads.
 * 
 * @param m_d  The matrix on device
 * @param jE   Column index of the entering variable
 * @param iL   Row index of the leaving variable
 * @param n    The maximum valid row index
 * @param d    The maximum valid column index
 */
__global__ void pivot_k(kilo::mpv m_d, u32 jE, u32 iL, u32 n, u32 d) {
	// Get row and column indices
	u32 i = blockIdx.x;
	u32 j = threadIdx.x;
	u32 t1 = tmp((i*d)+j+1, n, d);
	
	// Keep sign of M[iL,jE] in det (0)
	u32 Mij = elm(iL, jE, n, d);
	if ( i == 0 && j == 0 ) {
		if ( kilo::is_neg(m_d, Mij) ) { kilo::neg(m_d, 0); }
	}
	
	// Skip row/column of pivot
	if ( i >= iL ) ++i;
	if ( j >= jE ) ++j;
	
	// Rescale all other elements
	u32 Mi = elm(i, jE, n, d), Eij = elm(i, j, n, d);
	// M[i,j] = ( M[i,j]*M[iL,jE] - M[i,jE]*M[iL,j] )/det
	kilo::mul(m_d, t1, Eij, Mij);
	kilo::mul(m_d, Eij, Mi, elm(iL, j, n, d));
	kilo::sub(m_d, t1, Eij);
	kilo::div(m_d, Eij, t1, 0);
}

/**
 * Cleans up the tableau after a pivot. Fixes values in leaving row, entering column, and 
 * determinant, as well as swapping basic and cobasic variables. Intended to be private to 
 * cuda_tableau class; should be called with 1 block of blockSize threads.
 * 
 * @param blockSize  Block size of the invocation
 * @param m_d        The matrix on device
 * @param jE         Column index of the entering variable
 * @param iL         Row index of the leaving variable 
 * @param b_d        The basis buffer on device
 * @param c_d        The cobasis buffer on device
 * @param u_d        The device-side limb counter
 * @param n          The maximum valid row index
 * @param d          The maximum valid column index
 */
template<u32 blockSize>
__global__ void postPivot_k(kilo::mpv m_d, u32 jE, u32 iL, u32* b_d, u32* c_d, u32* u_d, 
                            u32 n, u32 d) {
	u32 tid = threadIdx.x;
	u32 Mij = elm(iL, jE, n, d);
	
	// Update used limb counter
	__shared__ u32 u_n[blockSize];
	u_n[tid] = 0;
	for (u32 i = tid; i <= (n+1)*(d+1); i += blockSize) {
		u32 u_i = kilo::size(m_d, i);
		if ( u_i > u_n[tid] ) u_n[tid] = u_i;
	}
	__syncthreads();
	
	// Reduce (with last warp unrolled)
	for (u32 s = blockSize >> 1; s > 32; s >>= 1) {
		if ( tid < s ) {
			if ( u_n[tid+s] > u_n[tid] ) u_n[tid] = u_n[tid+s];
		}
		
		__syncthreads();
	}
	if ( blockSize >= 64 && tid < 32 && u_n[tid+32] > u_n[tid] ) u_n[tid] = u_n[tid+32];
	if ( blockSize >= 32 && tid < 16 && u_n[tid+16] > u_n[tid] ) u_n[tid] = u_n[tid+16];
	if ( blockSize >= 16 && tid <  8 && u_n[tid+ 8] > u_n[tid] ) u_n[tid] = u_n[tid+ 8];
	if ( blockSize >=  8 && tid <  4 && u_n[tid+ 4] > u_n[tid] ) u_n[tid] = u_n[tid+ 4];
	if ( blockSize >=  4 && tid <  2 && u_n[tid+ 2] > u_n[tid] ) u_n[tid] = u_n[tid+ 2];
	// Combine last step of reduction with output
	if ( tid == 0 ) {
		if ( u_n[1] > u_n[0] ) *u_d = u_n[1];
		else *u_d = u_n[0];
	}
	
	// Recalculate pivot row/column
	if ( kilo::is_pos(m_d, Mij) ) {
		for (u32 j = tid; j <= d; j += blockSize) {
			kilo::neg(m_d, elm(iL, j, n, d));
		}
	} else {
		for (u32 i = tid; i <= n; i += blockSize) {
			kilo::neg(m_d, elm(i, jE, n, d));
		}
	}
	
	__syncthreads();
	
	if ( tid == 0 ) {
		// Reset pivot element, determinant
		kilo::swap(m_d, 0, Mij);
		if ( kilo::is_neg(m_d, 0) ) { kilo::neg(m_d, 0); }
		
		// Swap basic and cobasic variables
		u32 t = b_d[iL]; b_d[iL] = c_d[jE]; c_d[jE] = t;
	}
}

class cuda_tableau {
private:  //internal convenience functions
	
	/** Ensures at least a_n limbs are allocated in the device matrix */
	void ensure_limbs_d(u32 a_n) {
		if ( a_n > a_dl ) {
			m_d = kilo::expand_d(m_d, m_dl, a_dl, a_n);
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
	void ensure_limbs(u32 a_n) const {
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
		
		// Allocate limb count, output buffer, basis, cobasis, and matrix storage on device
		cudaMalloc((void**)&u_d, sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&o_d, sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
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
		
		// Allocate limb count, output buffer, basis, cobasis, and matrix storage on device
		cudaMalloc((void**)&u_d, sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&o_d, sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
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
		cudaFree(o_d); CHECK_CUDA_SAFE
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
			cudaMalloc((void**)&b_d, (n+1)*sizeof(u32)); CHECK_CUDA_SAFE
			cudaMalloc((void**)&c_d, (d+1)*sizeof(u32)); CHECK_CUDA_SAFE
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
		u32 enter = n+d+1, leave = 0, jE;
		
		// Find first cobasic variable with positive objective value on device
		posObj_k<32><<< 1, 32 >>>(m_d, c_d, o_d, n, d); CHECK_CUDA_SAFE
		cudaMemcpy(&enter, o_d, sizeof(u32), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		
		// If no increasing variables found, this is optimal
		if ( enter == n+d+1 ) return tableau_optimal;
		jE = col[enter];
		
		ensure_temp_space_d();  // Ensure enough space in device temporary variables
		
		// Find minimum ratio for entering variable, choosing good block size for coalescing
		if ( n < 128 ) {
			minRatio_k<32><<< 1,32 >>>(m_d, u_d, jE, b_d, o_d, n, d); CHECK_CUDA_SAFE
		} else if ( n < 256 ) {
			minRatio_k<64><<< 1,64 >>>(m_d, u_d, jE, b_d, o_d, n, d); CHECK_CUDA_SAFE
		} else if ( n < 512 ) {
			minRatio_k<128><<< 1,128 >>>(m_d, u_d, jE, b_d, o_d, n, d); CHECK_CUDA_SAFE
		} else if ( n < 1024 ) {
			minRatio_k<256><<< 1,256 >>>(m_d, u_d, jE, b_d, o_d, n, d); CHECK_CUDA_SAFE
		} else /* if ( n >= 1024 ) */ {
			minRatio_k<512><<< 1,512 >>>(m_d, u_d, jE, b_d, o_d, n, d); CHECK_CUDA_SAFE
		}
		cudaMemcpy(&leave, o_d, sizeof(u32), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
				
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
	 * @param enter      The index to enter the basis
	 * @param leave      The index to leave the basis
	 * @param has_space  Has space for the calculation been ensured (yes if ratioTest() was the 
	 *                   last call) [true]
	 */
	void doPivot(u32 enter, u32 leave, bool has_space = true) {
		u32 iL = row[leave];  // Leaving row
		u32 jE = col[enter];  // Entering column
		
		// Make sure enough space (generally done by ratioTest())
		if ( ! has_space ) ensure_temp_space_d();
		
		// Perform pivot on device
		pivot_k<<< n, d >>>(m_d, jE, iL, n, d); CHECK_CUDA_SAFE
		postPivot_k<128><<< 1, 128 >>>(m_d, jE, iL, b_d, c_d, u_d, n, d); CHECK_CUDA_SAFE
		
		// Fix row and column
		row[leave] = 0;
		row[enter] = iL;
		col[enter] = 0;
		col[leave] = jE;
	}

	/** Get a read-only matrix copy */
	const kilo::mpv& mat() const {
		ensure_limbs(a_dl);
		kilo::copy_dh(m, m_d, m_hl, a_dl);
		return m;
	}
	
private:  //class members
	u32 n;                ///< number of equations in tableau
	u32 d;                ///< dimension of underlying space
	
	u32* b_d;             ///< basis on device
	u32* c_d;             ///< cobasis on device
	u32* o_d;             ///< output parameter on device
	kilo::mpv m_d;        ///< tableau matrix on device
	
	mutable u32* b;       ///< host-side basis variable buffer
	mutable u32* c;       ///< host-side cobasis variable buffer
	mutable kilo::mpv m;  ///< host-side matrix buffer
	
	u32* row;             ///< row indices for variables
	u32* col;             ///< column indices for variables
	
	mutable u32 a_hl;     ///< number of limbs allocated for host matrix
	u32 a_dl;             ///< number of limbs allocated for device matrix
	u32 u_l;              ///< maximum number of limbs used for matrix
	u32* u_d;             ///< device storage for number of used limbs
	u32 m_dl;             ///< number of elements in the device matrix (includes temps)
	u32 m_hl;             ///< number of elements in the host matrix (excludes temps)
	
}; /* class cuda_tableau */

} /* namespace ksimplex */
