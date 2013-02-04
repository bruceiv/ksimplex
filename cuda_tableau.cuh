#ifndef _SUMP_CUDA_TABLEAU_CUH_
#define _SUMP_CUDA_TABLEAU_CUH_
/** GPU-based floating-point tableau for the "Simplex Using Multi-Precision" 
 *  (sump) project.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <algorithm>

#include <cuda.h>

#include "sump.hpp"
#include "sump_cuda.cuh"

namespace sump {
	
	/** Indexes into matrix.
	 *  
	 *  @param T		Element type of the matrix
	 *  @param m_d		The matrix to index into
	 *  @param i		The row index to retrieve
	 *  @param j		The column index to retrieve
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return a reference to the matrix value at the given row and column
	 */
	template<typename T>
	__device__ T& el(T* m_d, ind i, ind j, ind n, ind d) {
		return m_d[i*(d+1)+j];
	}
	
	/** Finds all the values in the objective row of the matrix with a 
	 *  positive coefficient. Will mark indices 1--d of the output index_list 
	 *  0 for not positive or 1 for positive. Intended to be private to 
	 *  cuda_tableau class; should be called with one block of d threads.
	 *  
	 *  @param T		Element type of the matrix
	 *  @param m_d		The matrix on device
	 *  @param buf_d	The output index list on device (should index up to at 
	 *  				least d)
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 */
	template<typename T>
	__global__ void posObj_d(T* m_d, index_list buf_d, ind n, ind d) {
		const T zero = element_traits< T >::zero;
		
		// Get column index
		ind j = threadIdx.x + 1;
		
		// Check value at this column
		buf_d[j] = ( el(m_d, 0, j, n, d) > zero ) ? 1 : 0;
	}
	
	/** Finds the leaving variable which most improves the objective for a 
	 *  given entering variable. Will return leaving variable in index 0 of 
	 *  the buffer index_list, 0 for unbounded. Intended to be private 
	 *  to cuda_tableau class; should be called with one block of 2^k threads, 
	 *  for some k.
	 *  
	 *  @param T			Element type of matrix
	 *  @param blockSize	Block size of the invocation
	 *  @param m_d			The matrix on device
	 *  @param jE			Column index of entering variable
	 *  @param b_d			Index buffer on device (should be set to basis 
	 *  					before invocation)
	 *  @param n			The maximum valid row index
	 *  @param d			The maximum valid column index
	 */
	template<typename T, ind blockSize>
	__global__ void minRatio_d(T* m_d, ind jE, index_list b_d, ind n, ind d) {
		const T zero = element_traits< T >::zero;
		
		// Column index for leaving variable which improves objective by 
		// maximum amount
		__shared__ ind leave[blockSize];
		// Minimum ratio
		__shared__ T minRatio[blockSize];
		
		ind tid = threadIdx.x;
		
		// First find min ratios for each thread
		leave[tid] = 0;
		minRatio[tid] = zero;
		
		for (ind iL = tid+1; iL <= n; iL += blockSize) {
			T t = el(m_d, iL, jE, n, d);
			// Negative value in entering column
			if ( t < zero ) {
				if ( leave[tid] == 0 ) {
					// First possible leaving variable
					leave[tid] = iL;
					minRatio[tid] = -el(m_d, iL, 0, n, d) / t;
				} else {
					// Test against previous best leaving value
					t = -el(m_d, iL, 0, n, d) / t;
					if ( t < minRatio[tid] 
						 || (t == minRatio[tid] && b_d[iL] < b_d[leave[tid]]) 
					   ) {
						leave[tid] = iL;
						minRatio[tid] = t;
					}
				}
			}
		}
		__syncthreads();
		
		// Reduce
		for (ind s = blockSize >> 1; s > 0; s >>= 1) {
			if ( tid < s ) {
				if ( leave[tid+s] != 0 
					 && (leave[tid] == 0 
					     || minRatio[tid+s] < minRatio[tid]
					     || (minRatio[tid+s] == minRatio[tid]
					         && b_d[leave[tid+s]] < b_d[leave[tid]])
					    ) 
				   ) {
					leave[tid] = leave[tid+s];
					minRatio[tid] = minRatio[tid+s];
				}
			}
			
			__syncthreads();
		}
		
		// Report minimum ratio
		b_d[0] = b_d[leave[0]];
	}
	
	/** Substitutes one equation for another as part of a pivot operation.
	 *  Redefines the leaving row of the tableau in terms of the entering 
	 *  variable. Intended to be private to cuda_tableau class; should be 
	 *  called with one block of d+1 threads, and followed by pivot_d().
	 *  
	 *  @param T		Element type of the matrix
	 *  @param m_d		The matrix on device
	 *  @param jE		The matrix column of the entering variable
	 *  @param iL		The matrix row of the leaving variable
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 */
	template<typename T>
	__global__ void subEqn_d(T* m_d, ind jE, ind iL, ind n, ind d) {
		const T one = element_traits< T >::one;
		
		// Get column index
		ind j = threadIdx.x;
		
		// Get scale factor
		T scale = -el(m_d, iL, jE, n, d);
		__syncthreads();
		
		// Subtract entering variable from both sides
		if ( j == jE ) { el(m_d, iL, j, n, d) = -one; }
		
		// Scale equation, if needed
		if ( scale != one ) {
			el(m_d, iL, j, n, d) /= scale;
		}
	}
	
	/** Pivots the tableau on the device. Substitutes equation iL (which, 
	 *  as this should be preceded by subEqn_d(), actually corresponds to 
	 *  jE) into all the remaining equations. Intended to be private to 
	 *  cuda_tableau class; should be called with n blocks of d+1 threads.
	 *  
	 *  @param T		Element type of the matrix
	 *  @param m_d		The matrix on device
	 *  @param jE		Column index of the entering variable
	 *  @param iL		Row index of the leaving variable
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 */
	template<typename T>
	__global__ void pivot_d(T* m_d, ind jE, ind iL, ind n, ind d) {
		const T zero = element_traits< T >::zero;
		
		// Get row and column indices
		ind i = blockIdx.x;
		if ( i >= iL ) ++i;
		ind j = threadIdx.x;
		
		// Get scale factor
		T scale = el(m_d, i, jE, n, d);
		__syncthreads();
		
		// Subtract entering variable from both sides
		if ( j == jE ) el(m_d, i, j, n, d) = zero;
		
		// Substitute with scaling
		el(m_d, i, j, n, d) += el(m_d, iL, j, n, d) * scale;
	}
	
	/** Implements the Tableau contract using floating-point arithmetic on GPU.
	 *  Keeps canonical data matrix on device, with a pre-allocated buffer 
	 *  for output on the host. Note that cuda_tableau is NOT rentrant.
	 *  
	 *  @param T		The floating-point type to use (should be one of 
	 *  				{float,double}. Note double may not work well on GPUs 
	 *  				with lower compute capability)
	 */
	template<typename T>
	class cuda_tableau {
	public:
		typedef T value_type;
		typedef T* vector_type;
		typedef T* matrix_type;
		
	private:
		typedef T value_type_d;
		
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
		 *  @param n	The number of equations in the tableau
		 *  @param d	The dimension of the underlying space
		 *  @param cob	The indices of the initial cobasis (should be sorted 
		 *  			in increasing order, cob[0] = 0 (the constant term))
		 *  @param bas	The indices of the initial basis (should be sorted in 
		 *  			increasing order, bas[0] = 0 (the objective))
		 *  @param mat	The matrix of the initial tableau (should be 
		 *  			organized such that the variable at bas[i] is in row 
		 *  			mat[i], and the variable at cob[j] is in column 
		 *  			mat[][j]. Note that the 0-row is for the objective 
		 *  			function, and the 0-column is for the constant terms)
		 */
		cuda_tableau(ind n, ind d, index_list cob, index_list bas, 
					  matrix_type mat) : n(n), d(d) {
			
			// Allocate host-side index lists and matrix buffer
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			buf = allocIndexList(n+d);
			
			// Allocate matrix and transfer buffer on device
			m_d = allocMat_d< value_type_d >(n, d);
			buf_d = allocIndexList_d(n+d);
			
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
			
			// Copy cobasis and column indices
			c[0] = 0;
			c_j = 0;
			for (j = 1; j <= d; ++j) {
				c[j] = cob[j];
				while ( c_j < cob[j] ) col[c_j++] = 0;
				col[c_j++] = j;
			}
			while ( c_j <= n+d ) col[c_j++] = 0;
			
			// Copy matrix directly to device
			copyMat_hd(m_d, mat, n, d);
		}
		
		/** Copy constructor.
		 *  
		 *  @param that		The tableau to copy
		 */
		cuda_tableau(const cuda_tableau< T >& that) : n(that.n), d(that.d) {
			
			// Allocate host storage for basis, cobasis, row, column, matrix, 
			// and temporary buffer
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			buf = allocIndexList(n+d);
			
			// Allocate device storage for matrix and temporary buffer
			m_d = allocMat_d< value_type_d >(n, d);
			buf_d = allocIndexList_d(n+d);
			
			// Copy basis, cobasis, row, and column on host
			copyIndexList(b, that.b, n);
			copyIndexList(c, that.c, d);
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			
			// Copy matrix directly on the GPU
			copyMat_d(m_d, that.m_d, n, d);
		}
		
		/** Destructor. */
		~cuda_tableau() {
			// Free host storage
			freeIndexList(b);
			freeIndexList(c);
			freeIndexList(row);
			freeIndexList(col);
			freeMat< value_type >(m);
			freeIndexList(buf);
			
			// Free device storage
			freeMat_d< value_type_d >(m_d);
			freeIndexList_d(buf_d);
		}
		
		/** Assignment operator.
		 *  
		 *  @param that		The tableau to assign to this one
		 */
		cuda_tableau< T >& operator= (const cuda_tableau< T >& that) {
			
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
				freeIndexList(buf);
				
				// Free device storage
				freeMat_d< value_type_d >(m_d);
				freeIndexList_d(buf_d);
				
				// Allocate new host storage
				b = allocIndexList(n);
				c = allocIndexList(d);
				row = allocIndexList(n+d);
				col = allocIndexList(n+d);
				m = allocMat< value_type >(n, d);
				buf = allocIndexList(n+d);
				
				// Allocate new device storage
				m_d = allocMat_d< value_type_d >(n, d);
				buf_d = allocIndexList_d(n+d);
			}
			
			// Copy basis, cobasis, row, and column on host
			copyIndexList(b, that.b, n);
			copyIndexList(c, that.c, d);
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			
			// Copy matrix directly on the GPU
			copyMat_d(m_d, that.m_d, n, d);
			
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
			ind enter = 0;
			
			ind j, jE;
			
			// Find all cobasic variables with positive objective value on 
			// device
			posObj_d<<< 1, d >>>(m_d, buf_d, n, d);
			
			// Copy list of positive objective cobasic variables to host and 
			// find first
			copyIndexList_dh(buf, buf_d, d);
			for (j = 1; j <= n+d; ++j) {
				jE = col[j];	// Get column index of variable j
				if ( jE != 0 && buf[jE] == 1 ) {
					enter = j;
					break;
				}
			}
			
			// If no increasing variables found, this is optimal
			if ( enter == 0 ) return tableau_optimal;
			
			// Copy basis variable list to device for minRatio
			copyIndexList_hd(buf_d, b, n);
			
			// Find minimum ratio for entering variable, choosing good block 
			// size for coalescing
			if ( n < 8 ) {
				minRatio_d< T, 1 ><<< 1, 1 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 32 ) {
				minRatio_d< T, 18 ><<< 1, 8 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 64 ) {
				minRatio_d< T, 16 ><<< 1, 16 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 128 ) {
				minRatio_d< T, 32 ><<< 1, 32 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 256 ) {
				minRatio_d< T, 64 ><<< 1, 64 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 512 ) {
				minRatio_d< T, 128 ><<< 1, 128 >>>(m_d, jE, buf_d, n, d);
			} else if ( n < 1024 ) {
				minRatio_d< T, 256 ><<< 1, 256 >>>(m_d, jE, buf_d, n, d);
			} else /* if ( n >= 1024 ) */ {
				minRatio_d< T, 512 ><<< 1, 512 >>>(m_d, jE, buf_d, n, d);
			}
			
			copyIndexList_dh(buf, buf_d, 0);
			ind leave = buf[0];
			
			// If no limiting variables found, this is unbounded
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
		void pivot(ind enter, ind leave) {
			ind iL = row[leave];	/**< leaving row */
			ind jE = col[enter];	/**< entering column */
			
			// Perform pivot on device
			subEqn_d<<< 1, d+1 >>>(m_d, jE, iL, n, d);
			pivot_d<<< n, d+1 >>>(m_d, jE, iL, n, d);
			
			// Fix basis, cobasis, row, and column 
			b[iL] = enter;
			c[jE] = leave;
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
		const index_list& basis() const { return b; }
		/** @return cobasic variables */
		const index_list& cobasis() const { return c; }
		
		/** @return underlying matrix for tableau. 0th row is objective 
		 *  		function, 0th column is constant coefficients. Otherwise, 
		 *  		row i corresponds to basis()[i], and column j corresponds 
		 *  		to cobasis()[j]  */
		const matrix_type& mat() const { 
			copyMat_dh(m, m_d, n, d);
			return m;
		}
	
	private:
		ind n;						/**< Number of equations in tableau */
		ind d;						/**< Dimension of underlying space */
		
		index_list b;				/**< Basis variables */
		index_list c;				/**< Cobasis variables */
		
		index_list row;				/**< Row indices for variables */
		index_list col;				/**< Column indices for variables */
		
		matrix_type m_d;			/**< Underlying matrix for tableau */
		
		mutable matrix_type m;		/**< Host-side buffer for matrix */
		
		mutable index_list buf_d;	/**< Device-side temporary buffer */
		mutable index_list buf;		/**< Host-side temporary buffer */
		
	}; /* class cuda_tableau */
	
}
#endif /* _SUMP_CUDA_TABLEAU_CUH_ */
