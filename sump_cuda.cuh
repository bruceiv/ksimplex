#ifndef _SUMP_SUMP_CUDA_CUH_
#define _SUMP_SUMP_CUDA_CUH_
/** Common header for CUDA programming in the "Simplex Using Multi-Precision" 
 *  (sump) project. Defines operations on the CUDA device for basic data types.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <cuda.h>
#include "safe_cuda.cuh"

#include "sump.hpp"

namespace sump {
	
	//////////////////////////////////////////////////////////////////////////
	// Index list operations
	//////////////////////////////////////////////////////////////////////////
	
	/** Allocates an index_list on the device. Should be freed with 
	 *  freeIndexList_d().
	 *  
	 *  @param n		The number of elements in the list (ignores empty 0th 
	 *  				index)
	 *  @return the new index list
	 */
	index_list allocIndexList_d(ind n) {
		ind* d_d;
		cudaMalloc((void**)&d_d, (n+1)*sizeof(ind)); CHECK_CUDA_SAFE
		return d_d;
	}
	
	/** Copies an index list from host to device. The destination list should 
	 *  be at least as long as the source list.
	 *  
	 *  @param d_d		The destination list
	 *  @param s		The source list
	 *  @param n		The length of the list (ignoring the empty 0-th index)
	 *  @return the destination list
	 */
	index_list copyIndexList_hd(index_list d_d, const index_list s, ind n) {
		cudaMemcpy(d_d, s, (n+1)*sizeof(ind), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		return d_d;
	}
	
	/** Copies an index list between two locations on the device. The 
	 *  destination list should be at least as long as the source list.
	 *  
	 *  @param d_d		The destination list
	 *  @param s_d		The source list
	 *  @param n		The length of the list (ignoring the empty 0-th index)
	 *  @return the destination list
	 */
	index_list copyIndexList_d(index_list d_d, const index_list s_d, ind n) {
		cudaMemcpy(d_d, s_d, (n+1)*sizeof(ind), 
				   cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		return d_d;
	}
	
	/** Copies an index list from device to host. The destination list should 
	 *  be at least as long as the source list.
	 *  
	 *  @param d_d		The destination list
	 *  @param s		The source list
	 *  @param n		The length of the list (ignoring the empty 0-th index)
	 *  @return the destination list
	 */
	index_list copyIndexList_dh(index_list d, const index_list s_d, ind n) {
		cudaMemcpy(d, s_d, (n+1)*sizeof(ind), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		return d;
	}
	
	/** Deallocates an index list on the device.
	 *  
	 *  @param l_d		The list to free
	 */
	void freeIndexList_d(index_list l_d) {
		cudaFree(l_d); CHECK_CUDA_SAFE
	}
	
	//////////////////////////////////////////////////////////////////////////
	// Matrix operations
	//////////////////////////////////////////////////////////////////////////
	
	/** Allocates a matrix on the device. Should be freed with freeMat_d<T>().
	 *  
	 *  @param T		The type of the elements
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return The pointer to the allocated memory
	 */
	template<typename T>
	T* allocMat_d(ind n, ind d) {
		T* dst_d;
		cudaMalloc((void**)&dst_d, (n+1)*(d+1)*sizeof(T)); CHECK_CUDA_SAFE
		return dst_d;
	}
	
	/** Copies a matrix from host to device.
	 *  
	 *  @param T		The type of the elements
	 *  @param dst_d	The matrix on the device to copy into
	 *  @param src		The matrix on the host to copy from
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return a pointer to the destination
	 */
	template<typename T>
	T* copyMat_hd(T* dst_d, const T* src, ind n, ind d) {
		cudaMemcpy(dst_d, src, (n+1)*(d+1)*sizeof(T), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		return dst_d;
	}
	
	/** Copies a matrix between two locations on the device.
	 *  
	 *  @param T		The type of the elements
	 *  @param dst_d	The matrix on the device to copy into
	 *  @param src_d	The matrix on the device to copy from
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return a pointer to the destination
	 */
	template<typename T>
	T* copyMat_d(T* dst_d, const T* src_d, ind n, ind d) {
		cudaMemcpy(dst_d, src_d, (n+1)*(d+1)*sizeof(T), 
				   cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
		return dst_d;
	}
	
	/** Copies a matrix from device to host.
	 *  
	 *  @param T		The type of the elements
	 *  @param dst		The matrix on the host to copy into
	 *  @param src_d	The matrix on the device to copy from
	 *  @param n		The maximum valid row index
	 *  @param d		The maximum valid column index
	 *  @return a pointer to the destination
	 */
	template<typename T>
	T* copyMat_dh(T* dst, const T* src_d, ind n, ind d) {
		cudaMemcpy(dst, src_d, (n+1)*(d+1)*sizeof(T), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		return dst;
	}
	
	/** Deallocates a matrix on the device.
	 *  
	 *  @param T		The type of the scalar
	 *  @param m		The matrix
	 */
	template<typename T>
	void freeMat_d(T* m) {
		cudaFree(m); CHECK_CUDA_SAFE
	}
}

#endif /* _SUMP_SUMP_CUDA_CUH_ */
