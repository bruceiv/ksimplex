#ifndef _SUMP_FLOAT_TABLEAU_HPP_
#define _SUMP_FLOAT_TABLEAU_HPP_
/** CPU-based tableau for the "Simplex Using Multi-Precision" (sump) project.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include "sump.hpp"


namespace sump {
	
	/** Implements the Tableau contract on CPU.
	 *  
	 *  @param T		The type to use (should be either floating-point or 
	 *  				rational)
	 */
	template<typename T>
	class simple_tableau {
	public:
		typedef T value_type;
		typedef typename element_traits< T >::vector vector_type;
		typedef typename element_traits< T >::matrix matrix_type;
	
	private:
		/** Indexes into matrix.
		 *  
		 *  @param i		The row index to retrieve
		 *  @param j		The column index to retrieve
		 *  @return a reference to the matrix value at the given row and column
		 */		
		value_type& el(ind i, ind j) { return m[i*(d+1)+j]; }
		
		/** Indexes into matrix.
		 *  
		 *  @param i		The row index to retrieve
		 *  @param j		The column index to retrieve
		 *  @return a reference to the matrix value at the given row and column
		 */		
		const value_type& el(ind i, ind j) const { return m[i*(d+1)+j]; }
		
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
		simple_tableau(ind n, ind d, index_list cob, index_list bas, 
					  matrix_type mat) : n(n), d(d) {
			
			// Allocate basis, cobasis, row and column storage
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			
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
			
			// Copy matrix
			copyMat< value_type >(m, mat, n, d);
		}
		
		/** Copy constructor.
		 *  
		 *  @param that		The tableau to copy
		 */
		simple_tableau(const simple_tableau< T >& that) : n(that.n), d(that.d) {
			
			// Allocate basis, cobasis, row, column, and matrix storage
			b = allocIndexList(n);
			c = allocIndexList(d);
			row = allocIndexList(n+d);
			col = allocIndexList(n+d);
			m = allocMat< value_type >(n, d);
			
			// Copy basis, cobasis, row, column, and matrix
			copyIndexList(b, that.b, n);
			copyIndexList(c, that.c, d);
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			copyMat< value_type >(m, that.m, n, d);
		}
		
		/** Destructor. */
		~simple_tableau() {
			// Free storage
			freeIndexList(b);
			freeIndexList(c);
			freeIndexList(row);
			freeIndexList(col);
			freeMat< value_type >(m);
		}
		
		/** Assignment operator.
		 *  
		 *  @param that		The tableau to assign to this one
		 */
		simple_tableau< T >& operator= (const simple_tableau< T >& that) {
			
			if ( n != that.n || d != that.d ) {
				// Reallocate memory, if needed
				n = that.n;
				d = that.d;
				
				freeIndexList(b);
				freeIndexList(c);
				freeIndexList(row);
				freeIndexList(col);
				freeMat< value_type >(m);
				
				b = allocIndexList(n);
				c = allocIndexList(d);
				row = allocIndexList(n+d);
				col = allocIndexList(n+d);
				m = allocMat< value_type >(n, d);
			}
			
			// Copy values from other tableau
			copyIndexList(b, that.b, n);
			copyIndexList(c, that.c, d);
			copyIndexList(row, that.row, n+d);
			copyIndexList(col, that.col, n+d);
			copyMat< value_type >(m, that.m, n, d);
			
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
			
			ind iL, j, jE;
			const T zero = element_traits< T >::zero;
			
			// Find first cobasic variable with positive objective value
			for (j = 1; j <= n+d; ++j) {
				jE = col[j];	// Get column index of variable j
				if ( jE != 0 && el(0, jE) > zero ) {
					enter = j;
					break;
				}
			}
			
			// If no increasing variables found, this is optimal
			if ( enter == 0 ) return tableau_optimal;
			
			// Look for leaving variable which improves objective by maximum 
			// amount
			ind leave = 0;
			T minRatio = zero;
			
			for (iL = 1; iL <= n; ++iL) {
				T t = el(iL, jE);
				// Negative value in entering column
				if ( t < zero ) {
					if ( leave == 0 ) {
						// First possible leaving variable
						leave = b[iL];
						minRatio = -el(iL, 0) / t;
					} else {
						// Test against previous leaving variable
						T ratio = -el(iL, 0) / t;
						if ( ratio < minRatio ) {
							leave = b[iL];
							minRatio = ratio;
						}
					}
				}
			}
			
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
			
			ind i, j;
			const T zero = element_traits< T >::zero;
			const T one = element_traits< T >::one;
			
			// Get scale factor
			T scale = -el(iL, jE);
			
			// Subtract entering variable from both sides
			el(iL, jE) = -one;
			
			// Scale equation, if needed
			if ( scale != one ) {
				for (j = 0; j <= d; ++j) { el(iL, j) /= scale; }
			}
			
			// Substitute into other equations
			for (i = 0; i <= n; ++i) {
				// Skip leaving (now entering) equation
				if ( i == iL ) continue;
				
				// Get scale factor
				scale = el(i, jE);
				
				// Subtract entering variable from both sides
				el(i, jE) = zero;
				
				// Substitute with scaling
				for (j = 0; j <= d; ++j) {
					el(i, j) += el(iL, j) * scale;
				}
			}
			
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
		const matrix_type& mat() const { return m; }
	
	private:
		ind n;			/**< Number of equations in tableau */
		ind d;			/**< Dimension of underlying space */
		
		index_list b;	/**< Basis variables */
		index_list c;	/**< Cobasis variables */
		
		index_list row;	/**< Row indices for variables */
		index_list col;	/**< Column indices for variables */
		
		matrix_type m;	/**< Underlying matrix for tableau */
	}; /* class simple_tableau */
	
}
#endif /* _SUMP_FLOAT_TABLEAU_HPP_ */
