#ifndef _SAFE_CUDA_CUH_
#define _SAFE_CUDA_CUH_

#include <iostream>
#include <cuda.h>

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

#endif /* _SAFE_CUDA_CUH_ */
