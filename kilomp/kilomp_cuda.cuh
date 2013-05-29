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
	limb*[1+alloc_l] v_h;
	
	//zero element values
	limb[n] l;
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
	limb*[1+a_l] d_h;
	cudaMemcpy(d_h, d_d, (1+a_l)*sizeof(limb*), cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
	
	// Copy limb arrays up to device
	for (u32 i = 0; i <= a_l; ++i) {
		cudaMemcpy(d_h[i], s[i], n*limb_size, cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
	}
}

} /* namespace kilo */

