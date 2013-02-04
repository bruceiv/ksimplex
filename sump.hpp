#ifndef _SUMP_SUMP_HPP_
#define _SUMP_SUMP_HPP_
/** Common header for the "Simplex Using Multi-Precision" (sump) project. 
 *  Defines `sump` namespace, basic data types, etc.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <cstring>
#include <limits>

namespace sump {
	/** Unsigned integer type for indices */
	typedef 
		unsigned long
		ind;
	
	/** List of indices type. 1-indexed (the value in 0th index should always 
	 *  be 0) */
	typedef
		ind*
		index_list;
	
	/** Allocates an index_list. Should be freed with freeIndexList().
	 *  
	 *  @param n	The number of elements in the list (ignores empty 0th 
	 *  			index)
	 *  @return the new index list
	 */
	index_list allocIndexList(ind n) {
		return new ind[n+1];
	}
	
	/** Copies an index list. The destination list should be at least as long
	 *  as the source list.
	 *  
	 *  @param d	The destination list
	 *  @param s	The source list
	 *  @param n	The length of the list (ignoring the empty 0-th index)
	 *  @return the destination list
	 */
	index_list copyIndexList(index_list d, index_list s, ind n) {
		return (index_list)memcpy(d, s, (n+1)*sizeof(ind));
	}
	
	/** Deallocates an index list.
	 *  
	 *  @param l	The list to free
	 */
	void freeIndexList(index_list l) {
		delete[] l;
	}
	
	/** Represents a simplex pivot. */
	struct pivot {
		ind enter;	/**< entering variable */
		ind leave;	/**< leaving variable */
		
		/** Default constructor.
		 *  
		 *  @param e		The entering variable
		 *  @param l		The leaving variable
		 */
		pivot(ind e, ind l) : enter(e), leave(l) {}
		
		/** Copy constructor.
		 *  
		 *  @param that		The pivot to copy
		 */
		pivot(const pivot& that) : enter(that.enter), leave(that.leave) {}
		
		/** Assignment operator.
		 *  
		 *  @param that		The pivot to copy
		 */
		pivot& operator= (const pivot& that) {
			enter = that.enter;
			leave = that.leave;
			
			return *this;
		}
		
		/** Equality operator.
		 *  
		 *  @param that		The pivot to test for equality
		 */
		bool operator== (const pivot& that) {
			return (enter == that.enter && leave == that.leave);
		}
		
		/** Inequality operator.
		 *  
		 *  @param that		The pivot to test for inequality
		 */
		bool operator!= (const pivot& that) {
			return (enter != that.enter || leave != that.leave);
		}
	};
	
	/** Special pivot value to indicate that the tableau is optimal */
	static const pivot tableau_optimal = pivot(0, 0);
	
	/** Special pivot value to indicate that the tableau is unbounded */
	static const pivot tableau_unbounded = pivot(
			std::numeric_limits<ind>::max(), std::numeric_limits<ind>::max());
		
	
	/** Possible types of numbers */
	enum element_type { integral, fractional, floating_point };
	
	/** Traits class for manipulating numeric values */
	template<typename T>
	struct element_traits {
		typedef T* vector;		/**< default vector type */
		typedef T* matrix;		/**< default matrix type */
		
		/** Numeric type of element */
		static const element_type elType = integral;
	};
	
	/** Traits class for manipulating numeric values */
	template<>
	struct element_traits<float> {
		typedef float* vector;
		typedef float* matrix;
		
		static const element_type elType = floating_point;
	};
	
	/** Traits class for manipulating numeric values */
	template<>
	struct element_traits<double> {
		typedef double* vector;		/**< default vector type */
		typedef double* matrix;		/**< default matrix type */
		
		static const element_type elType = floating_point;
	};
	
	/** Matrix allocator. Should be freed with freeMat<Mat>().
	 *  
	 *  @param T		The element type
	 *  @param n		The maximum valid row index (from 0)
	 *  @param d		The maximum valid column index (from 0)
	 */
	template<typename T>
	T* allocMat(ind n, ind d) {
		return new T[(n+1)*(d+1)];
	}
	
	/** Matrix Copier. Destination and source matrices will contain the same 
	 *  values afterward, but be otherwise unlinked.
	 *  
	 *  @param T		The element type
	 *  @param dst		The destination matrix
	 *  @param src		The source matrix
	 *  @param n		The maximum valid row index (from 0)
	 *  @param d		The maximum valid column index (from 0)
	 *  @return the destination matrix
	 */
	template<typename T>
	T* copyMat(T* dst, T* src, ind n, ind d) {
		//return (T*)memcpy(dst, src, (n+1)*(d+1)*sizeof(T));
		for (int i = 0; i < (n+1)*(d+1); ++i) dst[i] = src[i];
		return dst;
	}
	
	/** Matrix deallocator.
	 *  
	 *  @param T		The element type
	 *  @param m		The matrix to free
	 */
	template<typename T>
	void freeMat(T* m) {
		delete[] m;
	}
}
#endif /* _SUMP_SUMP_HPP_ */
