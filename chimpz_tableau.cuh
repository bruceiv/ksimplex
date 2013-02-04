#ifndef _SUMP_CHIMPZ_TABLEAU_CUH_
#define _SUMP_CHIMPZ_TABLEAU_CUH_
/** GPU-based integer tableau for the "Simplex Using Multi-Precision" (sump) 
 *  project.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <algorithm>

#include <cuda.h>
#include "safe_cuda.cuh"

#include "sump.hpp"
#include "sump_cuda.cuh"
#include "chimp/chimp.cuh"
#include "chimp/chimp_transfer.cuh"

namespace sump {

	/** Indexes into matrix.
	 *  
	 *  @param m_d		The matrix to index into
	 *  @param i		The row index to retrieve
	 *  @param j		The column index to retrieve
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return a reference to the matrix value at the given row and column
	 */
	__device__ chimp::chimpz& el(chimp::chimpz* m_d, ind i, ind j, 
								 ind n, ind d) {
		return m_d[i*(d+1)+j];
	}
	
	/** Finds the smallest index of a cobasic variable having a value in the 
	 *  objective row of the matrix with a positive coefficient. Will return 
	 *  this value in c_d[0] (or 0 for no such index). Intended to be private 
	 *  to cuda_tableau class; should be called with one block of 2^k threads, 
	 *  for some k.
	 *  
	 *  @param blockSize	Block size of the invocation
	 *  @param m_d			The matrix on device
	 *  @param c_d			The cobasis index list on device (should index up 
	 *  					to at least d), c_d[0] is the output value
	 *  @param n			The maximum valid row index
	 *  @param d			The maximum valid column index
	 */
	template<ind blockSize> 
	__global__ void posObj_d(chimp::chimpz* m_d, index_list c_d, ind n, ind d) {
		
		// Column index for first entering variable which improves objective
		__shared__ ind enter[blockSize];
		
		// Get column index
		ind tid = threadIdx.x;
		
		// Initialize reduction to value one higher than maximum
		enter[tid] = n+d+1;
		
		// Check value at this thread's columns
		if ( tid < d && el(m_d, 0, tid+1, n, d).sign() > 0 ) {
			enter[tid] = c_d[tid+1];
		}
		for (ind j = tid+blockSize+1; j <= d; j += blockSize) {
			if ( c_d[j] < enter[tid] && el(m_d, 0, j, n, d).sign() > 0 ) {
				enter[tid] = c_d[j];
			}
		}
		__syncthreads();
		
		// Reduce
		for (ind s = blockSize >> 1; s > 1; s >>= 1) {
			if ( tid < s ) {
				if ( enter[tid] > enter[tid+s] ) enter[tid] = enter[tid+s];
			}
			
			__syncthreads();
		}
		// Combine last step of reduction with output
		if ( tid == 0 ) {
			if ( enter[0] > enter[1] ) c_d[0] = enter[1];
			else if ( enter[0] == n+d+1 ) c_d[0] = 0;
			else c_d[0] = enter[0];
		}
	}
	
	/** Finds the leaving variable which most improves the objective for a 
	 *  given entering variable. Will return leaving variable in index 0 of 
	 *  the buffer index_list, 0 for unbounded. Intended to be private 
	 *  to cuda_tableau class; should be called with one block of 2^k threads, 
	 *  for some k.
	 *  
	 *  @param blockSize	Block size of the invocation
	 *  @param m_d			The matrix on device
	 *  @param jE			Column index of entering variable
	 *  @param b_d			Index buffer on device (should be set to basis 
	 *  					before invocation)
	 *  @param n			The maximum valid row index
	 *  @param d			The maximum valid column index
	 */
	template<ind blockSize>
	__global__ void minRatio_d(chimp::chimpz* m_d, ind jE, index_list b_d, 
							   ind n, ind d) {
		// Row index for leaving variable which improves objective by maximum 
		// amount
		__shared__ ind leave[blockSize];
		
		ind tid = threadIdx.x;
		
		// First find min ratios for each thread
		leave[tid] = 0;
		
		for (ind iL = tid+1; iL <= n; iL += blockSize) {
			chimp::chimpz denIL = -el(m_d, iL, jE, n, d);
			// Negative value in entering column
			if ( denIL.sign() > 0 ) {
				if ( leave[tid] == 0 ) {
					// First possible leaving variable
					leave[tid] = iL;
				} else {
					// Test against previous best leaving value
					ind oldIL = leave[tid];
					chimp::chimpz rat = 
						(el(m_d, iL, 0, n, d) * -el(m_d, oldIL, jE, n, d)) 
						- (el(m_d, oldIL, 0, n, d) * denIL);
					if ( rat.sign() < 0 
						 || (rat.sign() == 0 && b_d[iL] < b_d[oldIL]) 
					   ) {
						leave[tid] = iL;
					}
				}
			}
		}
		__syncthreads();
		
		// Reduce
		for (ind s = blockSize >> 1; s > 0; s >>= 1) {
			if ( tid < s ) {
				if ( leave[tid+s] != 0 ) {
					if ( leave[tid] == 0 ) {
						leave[tid] = leave[tid+s];
					} else {
						ind oldIL = leave[tid], thatIL = leave[tid+s];
						chimp::chimpz rat = 
							(el(m_d, thatIL, 0, n, d) 
								* -el(m_d, oldIL, jE, n, d)) 
							- (el(m_d, oldIL, 0, n, d) 
								* -el(m_d, thatIL, jE, n, d));
						if ( rat.sign() < 0 
							 || (rat.sign() == 0 && b_d[thatIL] < b_d[oldIL]) 
						   ) {
							leave[tid] = thatIL;
						}
					}
				}
			}
			
			__syncthreads();
		}
		
		// Report minimum ratio
		if ( tid == 0 ) {
			b_d[0] = b_d[leave[0]];
		}
	}
	
	/** Pivots the tableau on the device. Substitutes all values but those in 
	 *  the leaving row and entering column.  Intended to be private to 
	 *  cuda_tableau class; should be called with n blocks of d threads.
	 *  
	 *  @param m_d		The matrix on device
	 *  @param det_d	The determinant on device
	 *  @param jE		Column index of the entering variable
	 *  @param iL		Row index of the leaving variable
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 */
	__global__ void pivot_d(chimp::chimpz* m_d, chimp::chimpz* det_d, 
							ind jE, ind iL, ind n, ind d) {
		const chimp::chimpz zero(0);
		
		// Get row and column indices
		ind i = blockIdx.x;
		if ( i >= iL ) ++i;
		ind j = threadIdx.x;
		if ( j >= jE ) ++j;
		
		const chimp::chimpz Mij = el(m_d, iL, jE, n, d);
		chimp::chimpz det = *det_d;
		
		// Keep sign of Mij in det
		if ( Mij < zero ) det = -det;
		
		// Rescale all elements
		el(m_d, i, j, n, d) = 
			( (el(m_d, i, j, n, d) * Mij) 
			   - (el(m_d, i, jE, n, d) * el(m_d, iL, j, n, d))
			)/det;
	}
	
	/** Cleans up the tableau on device after a pivot. Fixes values in leaving 
	 *  row, entering column, and determinant, as well as swapping basic and 
	 *  cobasic variables. Intended to be private to cuda_tableau class; should 
	 *  be called with 1 block of blockSize threads.
	 *  
	 *  @param m_d			The matrix on device
	 *  @param det_d		The determinant on device
	 *  @param jE			Column index of the entering variable
	 *  @param iL			Row index of the leaving variable
	 *  @param n			The maximum valid row index
	 *  @param d			The maximum valid column index
	 *  @param blockSize	Number of threads in a block - should be as close 
	 *  					to n as practical
	 */
	__global__ void postPivot_d(chimp::chimpz* m_d, chimp::chimpz* det_d, 
							index_list b_d, index_list c_d, ind jE, ind iL, 
							ind n, ind d, int blockSize) {
		const chimp::chimpz zero(0);
		const chimp::chimpz Mij = el(m_d, iL, jE, n, d);
		
		if ( Mij.sign() > 0 ) {
			for (ind i = threadIdx.x; i <= d; i += blockSize) {
				if ( i != jE ) {
					el(m_d, iL, i, n, d) = -el(m_d, iL, i, n, d);
				} else {
					el(m_d, iL, i, n, d) = *det_d;
					*det_d = Mij;
					//swap basic and cobasic variables
					ind t = b_d[iL]; b_d[iL] = c_d[jE]; c_d[jE] = t;
				}
			}
		} else {
			for (ind i = threadIdx.x; i <= n; i += blockSize) {
				if ( i != iL ) {
					el(m_d, i, jE, n, d) = -el(m_d, i, jE, n, d);
				} else {
					if ( Mij.sign() == 0 ) {
						el(m_d, i, jE, n, d) = *det_d;
						*det_d = zero;
					} else {
						el(m_d, i, jE, n, d) = -(*det_d);
						*det_d = -Mij;
					}
					//swap basic and cobasic variables
					ind t = b_d[iL]; b_d[iL] = c_d[jE]; c_d[jE] = t;
				}
			}
		}
	}
	
	/** Implements the Tableau contract using exact integer arithmetic on GPU.
	 *  Keeps canonical data matrix on device, with a pre-allocated buffer 
	 *  for output on the host. Note that chimpz_tableau is NOT rentrant.
	 */
	class chimpz_tableau {
	public:
		typedef chimp::chimpz value_type;
		typedef chimp::chimpz* vector_type;
		typedef chimp::chimpz* matrix_type;
		
	private:
		typedef chimp::chimpz value_type_d;
		
		/** Creates a new pivot with the given indices. Resolves naming 
		 *  conflicts between pivot struct and pivot method.
		 *  
		 *  @param enter		The entering variable
		 *  @param leave		The leaving variable
		 *  @return the pivot with the given entering and leaving variables
		 */
		static pivot makePivot(ind enter, ind leave) { 
			return sump::pivot(enter, leave);
		}
		
	public:
		/** Default constructor.
		 *  
		 *  @param n		The number of equations in the tableau
		 *  @param d		The dimension of the underlying space
		 *  @param cob		The indices of the initial cobasis (should be 
		 *  				sorted in increasing order, cob[0] = 0 (the 
		 *  				constant term))
		 *  @param bas		The indices of the initial basis (should be sorted 
		 *  				in increasing order, bas[0] = 0 (the objective))
		 *  @param determ	Initial determinant value
		 *  @param mat		The matrix of the initial tableau (should be 
		 *  				organized such that the variable at bas[i] is in 
		 *  				row mat[i], and the variable at cob[j] is in column 
		 *  				mat[][j]. Note that the 0-row is for the objective 
		 *  				function, and the 0-column is for the constant 
		 *  				terms)
		 */
		chimpz_tableau(ind n, ind d, index_list cob, index_list bas, 
					   value_type determ, matrix_type mat) : n(n), d(d) {
			
			// Allocate host-side index list and matrix buffers
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			
			// Allocate basis, cobasis, matrix, and determinant on device
			b_d = allocIndexList_d(n);
			c_d = allocIndexList_d(d);
			m_d = chimp::createChimpzVecOnDevice((n+1)*(d+1));
			det_d = chimp::createChimpzOnDevice();
			
			ind i, j, r_i, c_j;
			
			// Copy basis and row indices
			b[0] = 0;
			r_i = 0;
			for (i = 1; i <= n; ++i) {
				b[i] = bas[i];
				while ( r_i < bas[i] ) row[r_i++] = 0;
				row[r_i++] = i;
			}
			while ( r_i <= n+d ) row[r_i++] = 0;
			copyIndexList_hd(b_d, b, n);
			
			// Copy cobasis and column indices
			c[0] = 0;
			c_j = 0;
			for (j = 1; j <= d; ++j) {
				c[j] = cob[j];
				while ( c_j < cob[j] ) col[c_j++] = 0;
				col[c_j++] = j;
			}
			while ( c_j <= n+d ) col[c_j++] = 0;
			copyIndexList_hd(c_d, c, d);
			
			// Copy matrix and determinant to device
			chimp::copyChimpzVecToDevice(m_d, mat, (n+1)*(d+1));
			chimp::copyChimpzToDevice(det_d, &determ);
		}
		
		/** Copy constructor.
		 *  
		 *  @param that		The tableau to copy
		 */
		chimpz_tableau(const chimpz_tableau& that) : n(that.n), d(that.d) {
			
			// Allocate host storage for basis, cobasis, row, column, matrix, 
			// and temporary buffer
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			
			// Allocate basis, cobasis, matrix, and determinant on device
			b_d = allocIndexList_d(n);
			c_d = allocIndexList_d(d);
			m_d = chimp::createChimpzVecOnDevice((n+1)*(d+1));
			det_d = chimp::createChimpzOnDevice();
			
			// Copy row and column on host
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			
			// Copy basis, cobasis, matrix, and determinant directly on the GPU
			copyIndexList_d(b_d, that.b_d, n);
			copyIndexList_d(c_d, that.c_d, d);
			chimp::copyChimpzVecOnDevice(m_d, that.m_d, (n+1)*(d+1));
			chimp::copyChimpzOnDevice(det_d, that.det_d);
		}
		
		/** Destructor. */
		~chimpz_tableau() {
			// Free host storage
			freeIndexList(b);
			freeIndexList(c);
			freeIndexList(row);
			freeIndexList(col);
			freeMat< value_type >(m);
			
			// Free device storage
			freeIndexList_d(b_d);
			freeIndexList_d(c_d);
			chimp::destroyChimpzVecOnDevice(m_d);
			chimp::destroyChimpzOnDevice(det_d);
		}
		
		/** Assignment operator.
		 *  
		 *  @param that		The tableau to assign to this one
		 */
		chimpz_tableau& operator= (const chimpz_tableau& that) {
			
			if ( n != that.n || d != that.d ) {
				// Reallocate memory, if needed
				n = that.n;
				d = that.d;
				
				// Free host storage
				freeIndexList(b);
				freeIndexList(c);
				freeIndexList(row);
				freeIndexList(col);
				freeMat< value_type >(m);
				
				// Free device storage
				freeIndexList(b_d);
				freeIndexList(c_d);
				chimp::destroyChimpzVecOnDevice(m_d);
				
				// Allocate new host storage
				b = allocIndexList(n);
				c = allocIndexList(d);
				row = allocIndexList(n+d);
				col = allocIndexList(n+d);
				m = allocMat< value_type >(n, d);
				
				// Allocate new device storage
				b_d = allocIndexList_d(n);
				c_d = allocIndexList_d(d);
				m_d = chimp::createChimpzVecOnDevice((n+1)*(d+1));
			}
			
			// Copy row and column on host
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			
			// Copy basis, cobasis, matrix, and determinant directly on the GPU
			copyIndexList_d(b_d, that.b_d, n);
			copyIndexList_d(c_d, that.c_d, d);
			chimp::copyChimpzVecOnDevice(m_d, that.m_d, (n+1)*(d+1));
			chimp::copyChimpzOnDevice(det_d, that.det_d);
			
			return *this;
		}
		
		/** Finds the next pivot using Bland's rule.
		 *  
		 *  @return The next pivot by Bland's rule, tableau_optimal if no such 
		 *  		pivot because the tableau is optimal, or tableau_unbounded 
		 *  		if no such pivot because the tableau is unbounded.
		 */
		pivot ratioTest() const {
			// Look for entering variable
			ind enter = 0, leave = 0;
			
			ind jE;
			
			// Find first cobasic variable with positive objective value on 
			// device
			posObj_d<8><<< 1, 8 >>>(m_d, c_d, n, d); CHECK_CUDA_SAFE
			
			// If no increasing variables found, this is optimal
			copyIndexList_dh(&enter, c_d, 0);
			if ( enter == 0 ) return tableau_optimal;
			jE = col[enter];
			
			// Find minimum ratio for entering variable, choosing good block 
			// size for coalescing
			if ( n < 8 ) {
				minRatio_d< 1 ><<< 1, 1 >>>(m_d, jE, b_d, 
											n, d); CHECK_CUDA_SAFE
			} else if ( n < 32 ) {
				minRatio_d< 8 ><<< 1, 8 >>>(m_d, jE, b_d, 
											 n, d); CHECK_CUDA_SAFE
			} else if ( n < 64 ) {
				minRatio_d< 16 ><<< 1, 16 >>>(m_d, jE, b_d, 
											  n, d); CHECK_CUDA_SAFE
			} else if ( n < 128 ) {
				minRatio_d< 32 ><<< 1, 32 >>>(m_d, jE, b_d, 
											  n, d); CHECK_CUDA_SAFE
			} else if ( n < 256 ) {
				minRatio_d< 64 ><<< 1, 64 >>>(m_d, jE, b_d, 
											  n, d); CHECK_CUDA_SAFE
			} else if ( n < 512 ) {
				minRatio_d< 128 ><<< 1, 128 >>>(m_d, jE, b_d, 
												n, d); CHECK_CUDA_SAFE
			} else if ( n < 1024 ) {
				minRatio_d< 256 ><<< 1, 256 >>>(m_d, jE, b_d, 
												n, d); CHECK_CUDA_SAFE
			} else /* if ( n >= 1024 ) */ {
				minRatio_d< 512 ><<< 1, 512 >>>(m_d, jE, b_d, 
												n, d); CHECK_CUDA_SAFE
			}
			
			// If no limiting variables found, this is unbounded
			copyIndexList_dh(&leave, b_d, 0);
			if ( leave == 0 ) return tableau_unbounded;
			
			// Return pivot
			return makePivot(enter, leave);
		}
		
		/** Pivots the tableau from one basis to another. The caller is 
		 *  responsible to ensure that this is a valid pivot (that is, the 
		 *  given entering variable is cobasic, leaving variable is basic, and 
		 *  coefficient of the entering variable in the equation defining the 
		 *  leaving variable is non-zero).
		 *  
		 *  @param enter		The index to enter the basis
		 *  @param leave		The index to leave the basis
		 */
		void doPivot(ind enter, ind leave) {
			ind iL = row[leave];	/**< leaving row */
			ind jE = col[enter];	/**< entering column */
			
			ind blockSize = ( n > 512 ) ? 512 : n;
			
			// Perform pivot on device
			pivot_d<<< n, d >>>(m_d, det_d, jE, iL, n, d); CHECK_CUDA_SAFE
			postPivot_d<<< 1, blockSize >>>(m_d, det_d, b_d, c_d, jE, iL, 
									  n, d, blockSize); CHECK_CUDA_SAFE
			
			// Fix row, and column 
			row[leave] = 0;
			row[enter] = iL;
			col[enter] = 0;
			col[leave] = jE;
		}
		
		/** @return number of equations in tableau */
		ind size() const { return n; }
		/** @return dimension of underlying space */
		ind dim() const { return d; }
		
		/** @return basic variables */
		const index_list& basis() const {
			copyIndexList_dh(b, b_d, n);
			return b;
		}
		/** @return cobasic variables */
		const index_list& cobasis() const {
			copyIndexList_dh(c, c_d, d);
			return c;
		}
		
		/** @return the determinant of the current matrix */
		const value_type& determinant() const {
			chimp::copyChimpzToHost(&det, det_d);
			return det;
		}
		
		/** @return underlying matrix for tableau. 0th row is objective 
		 *  		function, 0th column is constant coefficients. Otherwise, 
		 *  		row i corresponds to basis()[i], and column j corresponds 
		 *  		to cobasis()[j]  */
		const matrix_type& mat() const {
			chimp::copyChimpzVecToHost(m, m_d, (n+1)*(d+1));
			return m;
		}
	
	private:
		ind n;						/**< Number of equations in tableau */
		ind d;						/**< Dimension of underlying space */
		
		index_list b_d;				/**< Basis variables on device */
		index_list c_d;				/**< Cobasis variables on device */
		
		index_list row;				/**< Row indices for variables */
		index_list col;				/**< Column indices for variables */
		
		value_type_d* det_d;		/**< Determinant on device */
		
		matrix_type m_d;			/**< Underlying matrix for tableau */
		
		mutable matrix_type m;		/**< Host-side buffer for matrix */
		mutable index_list b;		/**< Host-side buffer for basis */
		mutable index_list c;		/**< Host-side buffer for cobasis */
		mutable value_type det;		/**< Host-side buffer for determinant */
		
	}; /* class cuda_tableau */
}
#endif /* _SUMP_CUDA_TABLEAU_CUH_ */
