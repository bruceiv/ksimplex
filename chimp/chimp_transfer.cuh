#ifndef _CHIMP_CHIMP_TRANSFER_CUH_
#define _CHIMP_CHIMP_TRANSFER_CUH_

/** Methods for transfering chimpz between device and host, as well as on 
 *  device.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <cuda.h>
#include "safe_cuda.cuh"

#include "chimp.cuh"

namespace chimp {
	
	/** Private to this namespace, call through createChimpzOnDevice() */
	__global__ void createChimpz_k(chimpz** v_d) {
		*v_d = new chimpz();
	}
	
	/** Private to this namespace, call through createChimpzVecOnDevice() */
	__global__ void createChimpzVec_k(chimpz** v_d, int len) {
		*v_d = new chimpz[len];
	}
	
	/** Creates a new chimpz on the device, returning a device pointer to it.
	 *  @return a newly allocated device pointer to a chimpz. Should be freed 
	 *  		by destroyChimpzOnDevice()
	 */
	chimpz* createChimpzOnDevice() {
		chimpz *v_h, **v_d;
		cudaMalloc((void**)&v_d, sizeof(chimpz*)); CHECK_CUDA_SAFE
		
		createChimpz_k<<< 1,1 >>>(v_d); CHECK_CUDA_SAFE
		cudaMemcpy(&v_h, v_d, sizeof(chimpz*), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		
		return v_h;
	}
	
	/** Creates a new chimpz vector on the device, returning a device pointer 
	 *  to it.
	 *  @param len		The length of the vector
	 *  @return a newly allocated device pointer to a chimpz vector of length 
	 *  		len. Should be freed by destroyChimpzVecOnDevice()
	 */
	chimpz* createChimpzVecOnDevice(int len) {
		chimpz *v_h, **v_d;
		cudaMalloc((void**)&v_d, sizeof(chimpz*)); CHECK_CUDA_SAFE
		
		createChimpzVec_k<<< 1,1 >>>(v_d, len); CHECK_CUDA_SAFE
		cudaMemcpy(&v_h, v_d, sizeof(chimpz*), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		
		return v_h;
	}
	
	
	/** Private to this namespace, call through destroyChimpzOnDevice() */
	__global__ void destroyChimpz_k(chimpz* v_d) {
		delete v_d;
	}
	
	/** Private to this namespace, call through destroyChimpzVecOnDevice() */
	__global__ void destroyChimpzVec_k(chimpz* v_d) {
		delete[] v_d;
	}
	
	/** Frees a chimpz created by createChimpzOnDevice().
	 *  @param v_d		A device pointer to the chimpz to free
	 */
	void destroyChimpzOnDevice(chimpz* v_d) {
		destroyChimpz_k<<< 1,1 >>>(v_d); CHECK_CUDA_SAFE
	}
	
	/** Frees a chimpz vector created by createChimpzOnDevice().
	 *  @param v_d		A device pointer to the vector to free
	 */
	void destroyChimpzVecOnDevice(chimpz* v_d) {
		destroyChimpzVec_k<<< 1,1 >>>(v_d); CHECK_CUDA_SAFE
	}
	
	/** Private to this namespace, call through copyChimpzToDevice() */
	__global__ void initChimpz_k(chimpz* v_d, int a, int u, chimpz::limb* l_d) {
		*v_d = chimpz(a, u, l_d);
	}
	
	/** Private to this namespace, call through copyChimpzVecToDevice() */
	__global__ void initChimpzVec_k(chimpz* v_d, int* a_d, int* u_d, 
								 chimpz::limb** l_d, int len) {
		for (int i = threadIdx.x; i < len; i += blockDim.x) {
			v_d[i] = chimpz(a_d[i], u_d[i], l_d[i]);
		}
	}
	
	/** Copies a host chimpz to device.
	 *  @param d_d		Device pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_h		Host pointer to source
	 */
	void copyChimpzToDevice(chimpz* d_d, chimpz* s_h) {
		chimpz& z = *s_h;
		int a = z.limbs();
		
		if ( a > 0 ) {
			chimpz::limb *l_d;
			cudaMalloc((void**)&l_d, a*chimpz::limb_size); CHECK_CUDA_SAFE
			cudaMemcpy(l_d, z.data(), a*chimpz::limb_size, 
					   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
			
			initChimpz_k<<< 1,1 >>>(d_d, a, z.used(), l_d); CHECK_CUDA_SAFE
			
			cudaFree(l_d); CHECK_CUDA_SAFE
		} else {
			initChimpz_k<<< 1,1 >>>(d_d, a, z.used(), NULL); CHECK_CUDA_SAFE
		}
	}
	
	/** Copies a host chimpz vector to device.
	 *  @param d_d		Device pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_h		Host pointer to source
	 *  @param len		Length of the vector to copy
	 */
	void copyChimpzVecToDevice(chimpz* d_d, chimpz* s_h, int len) {
		//allocate host and device side parameter arrays
		int a_h[len], *a_d;
		int u_h[len], *u_d;
		chimpz::limb *l_dp[len], **l_d;
		cudaMalloc((void**)&a_d, len*sizeof(int)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&u_d, len*sizeof(int)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&l_d, len*sizeof(chimpz::limb*)); CHECK_CUDA_SAFE
		
		//copy values into parameter arrays
		for (int i = 0; i < len; ++i) {
			chimpz& z = s_h[i];
			int zs = z.limbs();
			
			//copy parameters into host-side arrays
			a_h[i] = zs;
			u_h[i] = z.used();
			
			//allocate device side arrays for the limbs, then copy them
			if ( zs > 0 ) {
				cudaMalloc((void**)&(l_dp[i]), 
						   zs*chimpz::limb_size); CHECK_CUDA_SAFE
				cudaMemcpy(l_dp[i], z.data(), zs*chimpz::limb_size, 
						   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
			} else {
				l_dp[i] = NULL;
			}
		}
		
		//copy values up to device
		cudaMemcpy(a_d, a_h, len*sizeof(int), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(u_d, u_h, len*sizeof(int), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		cudaMemcpy(l_d, l_dp, len*sizeof(chimpz::limb*), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		
		//set up vector on device
		initChimpzVec_k<<< 1,64 >>>(d_d, a_d, u_d, l_d, len); CHECK_CUDA_SAFE
		
		//cleanup
		for (int i = 0; i < len; ++i) if ( l_dp[i] != NULL ) {
			cudaFree(l_dp[i]); CHECK_CUDA_SAFE
		}
		cudaFree(l_d); CHECK_CUDA_SAFE
		cudaFree(u_d); CHECK_CUDA_SAFE
		cudaFree(a_d); CHECK_CUDA_SAFE
	}
	
	/** Private to this namespace, call through copyChimpzOnDevice() */
	__global__ void copyChimpz_k(chimpz* d_d, chimpz* s_d) {
		*d_d = *s_d;
	}
	
	/** Private to this namespace, call through copyChimpzVecOnDevice() */
	__global__ void copyChimpzVec_k(chimpz* d_d, chimpz* s_d, int len) {
		for (int i = threadIdx.x; i < len; i += blockDim.x) {
			d_d[i] = s_d[i];
		}
	}
	
	/** Copies a device chimpz vector into another device chimpz vector.
	 *  @param d_d		Device pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_d		Device pointer to source
	 */
	void copyChimpzOnDevice(chimpz* d_d, chimpz* s_d) {
		copyChimpz_k<<< 1,1 >>>(d_d, s_d); CHECK_CUDA_SAFE
	}
	
	/** Copies a device chimpz vector into another device chimpz vector.
	 *  @param d_d		Device pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_d		Device pointer to source
	 *  @param len		Length of the vector to copy
	 */
	void copyChimpzVecOnDevice(chimpz* d_d, chimpz* s_d, int len) {
		copyChimpzVec_k<<< 1,64 >>>(d_d, s_d, len); CHECK_CUDA_SAFE
	}
	
	/** Private to this namespace, call through copyChimpzToHost() */
	__global__ void extractChimpzParams_k(chimpz* v_d, int* a_d, int* u_d) {
		chimpz& z = *v_d;
		*a_d = z.limbs();
		*u_d = z.used();
	}
	
	/** Private to this namespace, call through copyChimpzVecToHost() */
	__global__ void extractChimpzVecParams_k(chimpz* v_d, int* a_d, int* u_d, 
										  int len) {
		for (int i = threadIdx.x; i < len; i += blockDim.x) {
			chimpz& z = v_d[i];
			a_d[i] = z.limbs();
			u_d[i] = z.used();
		}
	}
	
	/** Private to this namespace, call through copyChimpzToHost() */
	__global__ void extractChimpzLimbs_k(chimpz* v_d, chimpz::limb* l_d) {
		chimpz& z = *v_d;
		int a = z.limbs();
		const chimpz::limb *s_d = z.data();
		
		for (int j = 0; j < a; ++j) l_d[j] = s_d[j];
	}
	
	/** Private to this namespace, call through copyChimpzVecToHost() */
	__global__ void extractChimpzVecLimbs_k(chimpz* v_d, chimpz::limb** l_d, 
										  int len) {
		for (int i = threadIdx.x; i < len; i += blockDim.x) {
			chimpz& z = v_d[i];
			int ai = z.limbs();
			if ( ai > 0 ) {
				chimpz::limb *d_d = l_d[i];
				const chimpz::limb *s_d = z.data();
				
				for (int j = 0; j < ai; ++j) d_d[j] = s_d[j];
			}
		}
	}
	
	/** Copies a device chimpz into a host chimpz.
	 *  @param d_h		Host pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_d		Device pointer to source
	 */
	void copyChimpzToHost(chimpz* d_h, chimpz* s_d) {
		//allocate host and device side parameter arrays
		int a, *a_d;
		int u, *u_d;
		cudaMalloc((void**)&a_d, sizeof(int)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&u_d, sizeof(int)); CHECK_CUDA_SAFE
		
		//get integer parameters from device
		extractChimpzParams_k<<< 1,1 >>>(s_d, a_d, u_d); CHECK_CUDA_SAFE
		cudaMemcpy(&a, a_d, sizeof(int), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		cudaMemcpy(&u, u_d, sizeof(int), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		
		if ( a > 0 ) {
			chimpz::limb l[a], *l_d;
			cudaMalloc((void**)&l_d, a*chimpz::limb_size); CHECK_CUDA_SAFE
			
			extractChimpzLimbs_k<<< 1,1 >>>(s_d, l_d); CHECK_CUDA_SAFE
			cudaMemcpy(l, l_d, a*chimpz::limb_size, 
					   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
			
			*d_h = chimpz(a, u, l);
			
			cudaFree(l_d); CHECK_CUDA_SAFE
		} else {
			*d_h = chimpz(a, u, NULL);
		}
		
		//cleanup
		cudaFree(u_d); CHECK_CUDA_SAFE
		cudaFree(a_d); CHECK_CUDA_SAFE
	}
	
	/** Copies a device chimpz vector into a host chimpz vector.
	 *  @param d_h		Host pointer to destination (should already be 
	 *  				allocated)
	 *  @param s_d		Device pointer to source
	 *  @param len		Length of the vector to copy
	 */
	void copyChimpzVecToHost(chimpz* d_h, chimpz* s_d, int len) {
		//allocate host and device side parameter arrays
		int a_h[len], *a_d;
		int u_h[len], *u_d;
		chimpz::limb *l_h[len], *l_dp[len], **l_d;
		cudaMalloc((void**)&a_d, len*sizeof(int)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&u_d, len*sizeof(int)); CHECK_CUDA_SAFE
		cudaMalloc((void**)&l_d, len*sizeof(chimpz::limb*)); CHECK_CUDA_SAFE
		
		//get integer parameters from device
		extractChimpzVecParams_k<<< 1,64 >>>(s_d, a_d, u_d, 
											 len); CHECK_CUDA_SAFE
		cudaMemcpy(a_h, a_d, len*sizeof(int), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		cudaMemcpy(u_h, u_d, len*sizeof(int), 
				   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
		
		//allocate device side limb arrays
		for (int i = 0; i < len; ++i) {
			int ai = a_h[i];
			if ( ai > 0 ) {
				cudaMalloc((void**)&(l_dp[i]), 
						   ai*chimpz::limb_size); CHECK_CUDA_SAFE
			} else {
				l_dp[i] = NULL;
			}
		}
		cudaMemcpy(l_d, l_dp, len*sizeof(chimpz::limb*), 
				   cudaMemcpyHostToDevice); CHECK_CUDA_SAFE
		
		//get limbs from device
		extractChimpzVecLimbs_k<<< 1,64 >>>(s_d, l_d, len); CHECK_CUDA_SAFE
		
		//copy limbs down and set up host-side vector
		for (int i = 0; i < len; ++i) {
			int ai = a_h[i];
			
			if ( ai > 0 ) {
				l_h[i] = new chimpz::limb[ai];
				cudaMemcpy(l_h[i], l_dp[i], ai*chimpz::limb_size, 
						   cudaMemcpyDeviceToHost); CHECK_CUDA_SAFE
				
				d_h[i] = chimpz(ai, u_h[i], l_h[i]);
				
				delete[] l_h[i];
			} else {
				d_h[i] = chimpz(ai, u_h[i], NULL);
			}
		}
		
		//cleanup
		for (int i = 0; i < len; ++i) if ( l_dp[i] != NULL ) {
			cudaFree(l_dp[i]); CHECK_CUDA_SAFE
		}
		cudaFree(l_d); CHECK_CUDA_SAFE
		cudaFree(u_d); CHECK_CUDA_SAFE
		cudaFree(a_d); CHECK_CUDA_SAFE
	}
	
} /* namespace chimp */

#endif /* CHIMP_CHIMP_TRANSFER_CUH_ */
