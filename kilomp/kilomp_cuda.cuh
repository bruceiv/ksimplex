#pragma once

#include <iostream>

#include <cuda.h>

#include "kilomp.cuh"

/**
 * Utility methods for using kilomp values on device.
 * 
 * @author Aaron Moss
 */

#ifdef DEBUG_CUDA

/** Checks safety of CUDA calls - use only through CHECK_CUDA_SAFE macro. */
void checkCudaSafe(const char* f, int l) {
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err ) {
		std::cerr << "\nCUDA Error " << f << ":" << l << ": [" << err << "]`" 
			<< cudaGetErrorString(err) << "'" << std::endl;
		exit(1);
	}
}
#define CHECK_CUDA_SAFE checkCudaSafe( __FILE__ , __LINE__ );

#else /* ifndef DEBUG_CUDA */

#define CHECK_CUDA_SAFE

#endif /* ifdef DEBUG_CUDA */

namespace kilo {

/**
 * Initializes a mp-vector on device.
 * @param n			The number of elements in the vector
 * @param alloc_l	The number of data limbs to initially allocate 
 * 					(default 4, must be non-zero)
 * @return a new mp-vector with all elements zeroed.
 */
__host__ mpv init_mpv_d(u32 n, u32 alloc_l = 4) {
	//allocate limb array pointers
	limb* v_h[1+alloc_l];
	
	//zero element values
	limb l[n];
	for (u32 j = 0; j < n; ++j) { l[j] = 0; }
	cudaMalloc((void**)&v_h[0], n*limb_size); CHECK_CUDA_SAFE
	cudaMemcpy(v_h[0], l, n*limb_size, cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
	
	//allocate data limbs
	for (u32 i = 1; i <= alloc_l; ++i) {
		cudaMalloc((void**)&v_h[i], n*limb_size); CHECK_CUDA_SAFE
	}
	
	//allocate and return matrix
	mpv v_d;
	cudaMalloc((void**)&v_d, (1+alloc_l)*sizeof(limb*)); CHECK_CUDA_SAFE 
	cudaMemcpy(v_d, v_h, (1+alloc_l)*sizeof(limb*), cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
	
	return v_d;
}

/**
 * Copies a host mp-vector into a device mp-vector.
 * @param d_d       The destination vector
 * @param s			The source vector
 * @param n			The number of elements in the source vector (the destination vector should be 
 *                  allocated for at least as many)
 * @param a_l		The number of data limbs allocated in the source vector (the destination vector 
 * 					should be allocated for at least as many)
 */
__host__ void copy_hd(mpv d_d, mpv s, u32 n, u32 a_l) {
	// Copy the device pointers for the limb arrays down
	limb* d_h[1+a_l];
	cudaMemcpy(d_h, d_d, (1+a_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	
	// Copy limb arrays up to device
	for (u32 i = 0; i <= a_l; ++i) {
		cudaMemcpy(d_h[i], s[i], n*limb_size, cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
	}
}

/**
 * Copies a device mp-vector into another device mp-vector.
 * @param d_d       The destination vector
 * @param s_d		The source vector
 * @param n			The number of elements in the source vector (the destination vector should be 
 *                  allocated for at least as many)
 * @param a_l		The number of data limbs allocated in the source vector (the destination vector 
 * 					should be allocated for at least as many)
 */
__host__ void copy_dd(mpv d_d, mpv s_d, u32 n, u32 a_l) {
	// Copy the device pointers for the limb arrays down
	limb* d_h[1+a_l];
	limb* s_h[1+a_l];
	cudaMemcpy(d_h, d_d, (1+a_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	cudaMemcpy(s_h, s_d, (1+a_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	
	// Copy limb arrays across on device
	for (u32 i = 0; i <= a_l; ++i) {
		cudaMemcpy(d_h[i], s_h[i], n*limb_size, cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
	}
}

/**
 * Copies a device mp-vector into a host mp-vector.
 * @param d	        The destination vector
 * @param s_d		The source vector
 * @param n			The number of elements in the source vector (the destination vector should be 
 *                  allocated for at least as many)
 * @param a_l		The number of data limbs allocated in the source vector (the destination vector 
 * 					should be allocated for at least as many)
 */
__host__ void copy_dh(mpv d, mpv s_d, u32 n, u32 a_l) {
	// Copy the device pointers for the limb arrays down
	limb* s_h[1+a_l];
	cudaMemcpy(s_h, s_d, (1+a_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	
	// Copy limb arrays across on device
	for (u32 i = 0; i <= a_l; ++i) {
		cudaMemcpy(d[i], s_h[i], n*limb_size, cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	}
}

/**
 * Expands a mp-vector on device
 * @param v_d		The vector
 * @param n			The number of elements in the vector
 * @param old_l		The number of limbs already allocated
 * @param alloc_l	The new number of limbs (must be greater than old_l)
 * @return the modified vector
 */
__host__ mpv expand_d(mpv v_d, u32 n, u32 old_l, u32 alloc_l) {
	// Allocate new mpv on device
	mpv w_d;
	cudaMalloc((void**)&w_d, (1+alloc_l)*sizeof(limb*)); CHECK_CUDA_SAFE 
	
	// Copy old limb pointers
	cudaMemcpy(w_d, v_d, (1+old_l)*sizeof(limb*), cudaMemcpyDeviceToDevice); CHECK_CUDA_SAFE
	
	// Allocate new data limbs
	limb* w_h[alloc_l-old_l];
	for (u32 i = 0; i < alloc_l-old_l; ++i) {
		cudaMalloc((void**)&w_h[i], n*limb_size); CHECK_CUDA_SAFE
	}
	
	// Copy new limb pointers to device
	cudaMemcpy(w_d+old_l+1, w_h, (alloc_l-old_l)*sizeof(limb*), cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
	
	//replace v with w
	cudaFree(v_d); CHECK_CUDA_SAFE
	return w_d;
}

/**
 * Frees a mp-vector on device
 * @param v_d		The vector
 * @param alloc_l	The number of limbs allocated
 */
__host__ void clear_d(mpv v_d, u32 alloc_l) {
	// Copy the device pointers for the limb arrays down
	limb* v_h[1+alloc_l];
	cudaMemcpy(v_h, v_d, (1+alloc_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	
	// Free device arrays
	for (u32 i = 0; i <= alloc_l; ++i) { cudaFree(v_h[i]); CHECK_CUDA_SAFE }
	cudaFree(v_d); CHECK_CUDA_SAFE
}

} /* namespace kilo */

