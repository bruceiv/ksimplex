#pragma once

/** 
 * Main header for Kilo Multi-Precision Integer project.
 * Defines basic type (interleaved array of mp-ints), and operations on elements of that array.
 * 
 * The algorithms contained herein are largely derived from Knuth's Art of Programming; all but the 
 * division algorithm should be quite straightforward though.
 * 
 * @author Aaron Moss
 */

#include <cstdlib>

#include "fixed_width.hpp"

#ifdef __CUDA_ARCH__
#define DEVICE_HOST __device__ __host__
#define DEVICE_ONLY __device__
#define HOST_ONLY __host__
#else
#define DEVICE_HOST 
#define DEVICE_ONLY 
#define HOST_ONLY 
#endif

namespace kilo {
	
/**
 * A vector of multi-precision integers.
 * 
 * For mpv M, M[i][j] is the i'th limb of the j'th element in the 
 * vector. The magnitude of the 0th limb is the number of limbs used in 
 * an element (not counting the 0th limb), while the sign of the 0th 
 * limb is the sign of the element. The 1st limb contains the least 
 * significant 32 bits of the element, the 2nd limb the next-least 
 * significant 32 bits, and so forth.
 */
typedef u32** mpv;

/** Alias for limb type */
typedef u32 limb;
/** Size of a limb in bytes. */
static const u32 limb_size = 4;

/** @return how many limbs does the i'th element of v use */
DEVICE_HOST inline u32 size(const mpv v, u32 i) {
	return abs(static_cast<s32>(v[0][i]));
}

/** @return what is the sign of the i'th element of v */
DEVICE_HOST inline s32 sign(const mpv v, u32 i) {
	s32 u = static_cast<s32>(v[0][i]);
	return ( u > 0 ) - ( u < 0 );
}

/** @return do the i'th and j'th elements of v have the same sign */
DEVICE_HOST inline bool same_sign(const mpv v, u32 i, u32 j) {
	//test that the high bits of the sign fields are the same
	return (((v[0][i] ^ v[0][j]) & 0x80000000) == 0);
}

/** @return is the i'th element of v zero? */
DEVICE_HOST inline bool is_zero(const mpv v, u32 i) {
	return v[0][i] == 0;
}

/** @return is the i'th element of v positive? */
DEVICE_HOST inline bool is_pos(const mpv v, u32 i) {
	return static_cast<s32>(v[0][i]) > 0;
}

/** @return is the i'th element of v negative? */
DEVICE_HOST inline bool is_neg(const mpv v, u32 i) {
	return static_cast<s32>(v[0][i]) < 0;
}

/** @return cond ? -x : x */
DEVICE_HOST inline u32 neg_if(u32 x, bool cond) {
	return (x ^ -cond) + cond;
}

/**
 * Initializes a mp-vector.
 * @param n			The number of elements in the vector
 * @param alloc_l	The number of data limbs to initially allocate 
 * 					(default 4, must be non-zero)
 * @return a new mp-vector with all elements zeroed.
 */
DEVICE_HOST mpv init_mpv(u32 n, u32 alloc_l = 4) {
	//allocate limb array pointers
	mpv v = (mpv)malloc((1+alloc_l)*sizeof(limb*));
	
	//zero element values
	limb* l = (limb*)malloc(n*limb_size);
	for (u32 j = 0; j < n; ++j) { l[j] = 0; }
	v[0] = l;
	
	//allocate data limbs
	for (u32 i = 1; i <= alloc_l; ++i) {
		l = (limb*)malloc(n*limb_size);
		v[i] = l;
	}
	
	return v;
}

/**
 * Copies one mp-vector into another.
 * @param d			The destination vector
 * @param s			The source vector
 * @param n			The number of elements in the source vector (the destination vector should be 
 *                  allocated for at least as many)
 * @param a_l		The number of data limbs allocated in the source vector (the destination vector 
 * 					should be allocated for at least as many)
 */
DEVICE_HOST void copy(mpv d, mpv s, u32 n, u32 a_l) {
	for (u32 i = 0; i <= a_l; ++i) for (u32 j = 0; j < n; ++j) { d[i][j] = s[i][j]; }
}

/**
 * Expands a mp-vector
 * @param v			The vector
 * @param n			The number of elements in the vector
 * @param old_l		The number of limbs already allocated
 * @param alloc_l	The new number of limbs (must be greater than old_l)
 * @return the modified vector
 */
DEVICE_HOST mpv expand(mpv v, u32 n, u32 old_l, u32 alloc_l) {
	//re-allocate limb array pointers
	mpv w = (mpv)malloc((1+alloc_l)*sizeof(limb*));
	
	//copy old limb pointers
	u32 i = 0;
	for (; i <= old_l; ++i) { w[i] = v[i]; }
	//allocate new data limbs
	limb* l;
	for (; i <= alloc_l; ++i) {
		l = (limb*)malloc(n*limb_size);
		w[i] = l;
	}
	
	//replace v with w
	free(v);
	return w;
}

/**
 * Frees a mp-vector
 * @param v			The vector
 * @param alloc_l	The number of limbs allocated
 */
DEVICE_HOST void clear(mpv v, u32 alloc_l) {
	for (u32 i = 0; i <= alloc_l; ++i) { free(v[i]); }
	free(v);
}

/**
 * Copies one value into another slot
 * @param v			The vector
 * @param i			The slot to copy to
 * @param j			The slot to copy from
 */
DEVICE_HOST void assign(mpv v, u32 i, u32 j) {
	u32 nl = size(v, j);
	for (u32 k = 0; k <= nl; ++k) { v[k][i] = v[k][j]; }
}

/** Swaps the i'th and j'th elements of v */
DEVICE_HOST void swap(mpv v, u32 i, u32 j) {
	u32 nl = size(v, i);
	u32 nj = size(v, j);
	if ( nj > nl ) { nl = nj; }
	
	limb t;
	for (u32 k = 0; k <= nl; ++k) { t = v[k][i]; v[k][i] = v[k][j]; v[k][j] = t; }
}

/** Negates the i'th element of v */
DEVICE_HOST void neg(mpv v, u32 i) {
	v[0][i] = -(static_cast<s32>(v[0][i]));
}

/** Zeros the i'th element of v */
DEVICE_HOST void set_zero(mpv v, u32 i) {
	v[0][i] = 0;
}

/** Sets the i'th element of v to 1 (assumes there are data limbs allocated) */
DEVICE_HOST void set_one(mpv v, u32 i) {
	v[0][i] = 1; v[1][i] = 1;
}

/** @return is abs(i'th element of v) >= abs(j'th element of v) */
DEVICE_HOST bool cmpu(mpv v, u32 i, u32 j) {
	u32 i_len = size(v, i), j_len = size(v, j);
	
	//different lengths
	if ( i_len > j_len ) return true;
	else if ( i_len < j_len ) return false;
	
	//same length
	for (u32 k = i_len; k > 0; --k) {
		if ( v[k][i] == v[k][j] ) continue;
		else if ( v[k][i] > v[k][j] ) return true;
		else /* if ( v[k][i] < v[k][j] ) */ return false;
	}
	
	//elements are equal
	return true;
}

/** 
 * Compares i'th and j'th elements of v.
 * @return 0 for equal, 1 for i'th > j'th, -1 for i'th < j'th
 */
DEVICE_HOST s32 cmp(mpv v, u32 i, u32 j) {
	//compare number of limbs (signed)
	s32 l_d = static_cast<s32>(v[0][i]) - static_cast<s32>(v[0][j]);
	if ( l_d > 0 ) {  //i'th is more positive or less negative by at least a limb
		return 1;
	} else if ( l_d < 0 ) {  //i'th is less positive or more negative by at least a limb
		return -1;
	}
	
	//compare magnitudes - the values have the same sign and have the same number of limbs, but we 
	// need to account properly for the sign (greater magnitude negative is less-than, greater 
	// magnitude positive is greater-than)
	for (u32 k = size(v, i); k > 0; --k) {
		if ( v[k][i] == v[k][j] ) continue;
		else if ( v[k][i] > v[k][j] ) return sign(v, i);
		else /* if ( v[k][i] < v[k][j] ) */ return -sign(v, i);
	}
	
	//elements are equal
	return 0;
}

namespace {
/** 
 * Addition function - (r'th element of v) = (i'th element of v) + (j'th element of v).
 * @param v		The vector
 * @param r		The index of the result (may alias i or j)
 * @param i		The index of the first operand (if fewer limbs than n, the extras should be 
 * 				zeroed)
 * @param j		The index of the second operand
 * @param n		Number of limbs in v[j]
 * @return the carry bit from the last addition
 */
DEVICE_HOST static u32 add_l(mpv v, u32 r, u32 i, u32 j, u32 n) {
	u32 c = 0;
	for (u32 k = 1; k <= n; ++k) {
		u32 t = v[k][i] + v[k][j];                 //overflow iff v[k][i] > t or v[k][j] > t
		v[k][r] = t + c;                           //overflow iff t = 2**32 - 1 and c = 1
		c = (t < v[k][i]) | (c & (v[k][r] == 0));  //check overflow
	}
	return c;
}

/**
 * Subtraction function - (r'th element of v) = (i'th element of v) - (j'th element of v).
 * @param v		The vector
 * @param r		The index of the result (may alias i or j) (if fewer limbs than n, the extras 
 * 				should be zeroed)
 * @param i		The index of the first operand
 * @param j		The index of the second operand
 * @param n		Number of limbs in v[j]
 * @return the borrow bit from the last subtraction
 */
DEVICE_HOST static u32 sub_l(mpv v, u32 r, u32 i, u32 j, u32 n) {
	u32 c = 1;                                  // not-borrow - 1 for no borrow, 0 for borrow
	for (u32 k = 1; k <= n; ++k) {
		u32 t = v[k][i] + c;                    // add 1 to v[k][i] if not borrowing
		v[k][r] = t - v[k][j];                  // subtract v[k][j] from modified v[k][i]
		v[k][r] -= 1;                           // perform borrow
		c = ( v[k][r] <= t ) | (c & (t == 0));  // recalculate not-borrow flag
		                                        // if result is <= t, didn't underflow on subtract
		                                        // if no borrow and t == 0, v[i][j] == MAX, can't
		                                        // underflow 
	}
	return !c;
}

/** 
 * Multiplication function - (r'th element of v) = (i'th element of v) * (j'th element of v)
 * @param v		The vector
 * @param r		The index of the result (should be initially zeroed) (may NOT alias i or j)
 * @param i		The index of the first operand
 * @param j		The index of the second operand
 * @param n		The number of limbs in the i'th element of v
 * @param m		The number of limbs in the j'th element of v
 */
DEVICE_HOST static void mul_l(mpv v, u32 r, u32 i, u32 j, u32 n, u32 m) {
	for (u32 k = 0; k < m; ++k) {
		u32 c = 0;
		for (u32 l = 0; l < n; ++l) {
			u64 i_l = v[l+1][i], j_k = v[k+1][j], r_kl = v[k+l+1][r];
			
			//We know uv won't overflow as (taking n = 2**32), (n-1)**2 + 2(n-1) = n**2 - 1.
			// That is, we can multiply any two limbs, and add any two other limbs, the result will 
			// fit in a double-wide space.
			r_kl += c;
			u64 uv = (i_l * j_k) + r_kl;
			
			v[k+l+1][r] = static_cast<u32>(uv & 0xFFFFFFFF);  //put low-order bits in output
			c = static_cast<u32>(uv >> 32);                   //put high-order bits in carry
		}
		v[k+n+1][r] = c;  //store last carry in result
	}
}

/**
 * Single-limb division function - (r'th element of v) = (i'th element of v) / y
 * @param v		The vector
 * @param r		The index of the result
 * @param i		The index of the dividend
 * @param y		The divisior
 * @param n		The number of limbs in the i'th element of v (should be >= 1)
 * @return the remainder of the division
 */
DEVICE_HOST static limb div1_l(mpv v, u32 r, u32 i, limb y, u32 n) {
	u32 c = 0;
	u64 t;
	for (u32 k = n; k > 0; --k) {
		t = c; t <<= 32; t |= v[k][i];  //setup two limb temporary
		v[k][r] = (t / y);              //divide through
		c = (t % y);                    //propegate remainder
	}
	return c;
}

/**
 * Left-shift function - (i'th element of v) <<= s
 * @param v		The vector
 * @param i		The index of the value to shift
 * @param s		The amount to shift by
 * @param n		The number of limbs in the i'th element of v
 */
DEVICE_HOST static void lsh_l(mpv v, u32 i, u32 s, u32 n) {
	u32 s_l = s >> 5;    // (s / 32) -- number of limbs to shift
	u32 s_b = s & 0x1F;  // (s % 32) -- number of bits to shift
	
	u32 u_s = 32 - s_b;  // amount to "un-shift" high bits
	u32 mask = ((1 << s_b) - 1) << u_s;  //mask for high bits
	
	u32 prev, next;
	prev = next = v[n][i];
	//set new high limb
	v[n+s_l+1][i] = (next & mask) >> u_s;
	//shift remaining limbs
	for (u32 k = n; k > 1; --k) {
		next = v[k-1][i];
		//get low order bits of current limb, then high order bits of next limb back
		v[k+s_l][i] = (prev << s_b) | ((next & mask) >> u_s);
		prev = next;
	}
	//set new low limb
	v[s_l+1][i] = prev << s_b;
	
	//zero low-order limbs
	for (u32 k = s_l; k > 0; --k) { v[k][i] = 0; }
}

/**
 * Right-shift function - (i'th element of v) >>= s
 * @param v		The vector
 * @param i		The index of the value to shift
 * @param s		The amount to shift by
 * @param n		The number of limbs in the i'th element of v
 */
 DEVICE_HOST static void rsh_l(mpv v, u32 i, u32 s, u32 n) {
	u32 s_l = s >> 5;    // (s / 32) -- number of limbs to shift
	u32 s_b = s & 0x1F;  // (s % 32) -- number of bits to shift
	
	u32 u_s = 32 - s_b;  //amount to "un-shift" low bits
	u32 mask = (1 << s_b) - 1;  //mask for low bits
	
	u32 prev, next;
	prev = v[s_l+1][i];
	//right shift limbs
	for (u32 k = s_l+1; k < n; ++k) {
		next = v[k+1][i];
		v[k-s_l][i] = ((next & mask) << u_s) | (prev >> s_b);
		prev = next;
	}
	//set new high limb
	v[n-s_l][i] = prev >> s_b;
	
	//zero high-order limbs
	for (u32 k = n-s_l+1; k <= n; ++k) { v[k][i] = 0; }
 }

/** 
 * Multi-limb division function - (r'th element of v) = (i'th element of v) / (j'th element of v).
 * @param v		The vector
 * @param r		The index of the result (should be initially zeroed) (may NOT alias i or j)
 * @param i		The index of the dividend - will store remainder after computation
 * @param j		The index of the divisor
 * @param n		The number of limbs in the i'th element of v (should be >= m)
 * @param m		The number of limbs in the j'th element of v (should be >= 2, use div1_l otherwise)
 */
DEVICE_HOST static void divn_l(mpv v, u32 r, u32 i, u32 j, u32 n, u32 m) {
	u32 o = n - m;
	
	//normalize there is a 1 in the high bit of the j'th element of v
	
	//d to contain the index of the current high bit of the j'th element of v
	u32 h = v[m][j];
	u32 d = (h > 0xFFFF) << 4; h >>= d;  //account for high bit in top 16 bits
	u32 s = (h > 0xFF  ) << 3; h >>= s; d |= s;  //         ... in remaining top 8 bits
	    s = (h > 0xF   ) << 2; h >>= s; d |= s;  //         ... in remaining top 4 bits
	    s = (h > 0x3   ) << 1; h >>= s; d |= s;  //         ... in remaining top 2 bits
	                                    d |= (h >> 1);  //  ... in remaining bit
	d = 31 - d;  //how many bits to shift for normalization
	
	//normalize i'th and j'th elements by given amount
	if ( d > 0 ) {
		lsh_l(v, i, d, n);
		lsh_l(v, j, d, m);
	}
	
	//divide first m digits of the dividend by the divisor
	for (u32 k = o; true; --k) {
		//determine trial quotient qh
		u32 qh, rh;
		u64 t1, t2, t3;
		if ( v[k+m+1][i] == v[m][j] ) {
			qh = 0xFFFFFFFF;
		} else {
			t1 = v[k+m+1][i]; t1 <<= 32; t1 |= v[k+m][i];
			qh = t1 / v[m][j];
			rh = t1 % v[m][j];
			
			t2 = v[m-1][j]; t2 *= qh;
			t3 = rh; t3 <<= 32; t3 |= v[k+m-1][i];
			while ( t2 > t3 ) {
				//update trial quotient if needed - will run at most twice (?)
				--qh;
				if ( rh + v[m][j] < rh ) break;
				rh += v[m][j];
				
				t2 = v[m-1][j]; t2 *= qh;
				t3 = rh; t3 <<= 32; t3 |= v[k+m-1][i];
			}
		}
		
		//replace dividend by remainder of division by qh * divisor
		u32 b = 0, c = 0;
		for (u32 l = 1; l <= m; ++l) {
			t1 = qh; t1 *= v[l][j]; t1 += b;
			c = (t1 & 0xFFFFFFFF);
			b = (t1 >> 32);
			
			if (v[k+l][i] < c) ++b;
			
			v[k+l][i] -= c;
		}
		
		if ( b > v[k+m+1][i] ) {
			//check case where qh was 1 too big (if so, add a divisor back in)
			v[k+m+1][i] -= b;
			qh--;
			
			c = 0;
			for (u32 k_a = k+1; k_a <= m+k; ++k_a) {
				u32 t_a = v[k_a][i] + v[k_a][j];
				v[k_a][i] = t_a + c;
				c = (t_a < v[k_a][i]) | (c & (v[k_a][i] == 0));
			}
			//ignore final carry, it cancels the borrow from earlier
		} else {
			v[k+m+1][i] -= b;
		}
		
		//store final quotient
		v[k+1][r] = qh;
		
		if ( k == 0 ) break;
	}
	
	//un-normalize i'th and j'th elements
	if ( d > 0 ) {
		rsh_l(v, i, d, n+1);
		rsh_l(v, j, d, m);
	}
}

} /* unnamed */

/** 
 * Adds the j'th element of v to the i'th element of v.
 * Assumes there are enough limbs allocated in v to hold the result of the addition; 
 * max{size(v, i), size(v, j)} + 1 will do.
 * @return the number of limbs used by the i'th element of v after addition
 */
DEVICE_HOST static u32 add(mpv v, u32 i, u32 j) {
	
	if ( same_sign(v, i, j) ) { //same sign, add
		
		u32 len = size(v, i), that_len = size(v, j);
	
		//ensure extra limbs are zeroed
		u32 k = len+1;
		for (; k <= that_len; ++k) { v[k][i] = 0; }
		
		//add
		u32 c = add_l(v, i, i, j, that_len);
		//propegate carry
		k = that_len+1;
		for (; k <= len; ++k) {
			v[k][i] += c;
			c &= (v[k][i] == 0);
		}
		v[k][i] = c;
		
		//reset length -- k, accounting for the possibility of no carry, with the previous sign
		u32 r = k - (c == 0);
		v[0][i] = neg_if(r, is_neg(v, i));
		return r;
		
	} else {  //opposite signs, subtract
		
		u32 g, l;  //greater and lesser magnitude indices
		if ( cmpu(v, i, j) ) {
			g = i; l = j;
		} else {
			g = j; l = i;
		}
		u32 g_len = size(v, g), l_len = size(v, l);
		
		//subtract
		u32 b = sub_l(v, i, g, l, l_len);
		//propegate borrow (will not overflow, by construction)
		for (u32 k = l_len+1; k <= g_len; ++k) {
			v[k][i] = v[k][g] - b;
			b &= (v[k][i] == 0xFFFFFFFF);
		}
		
		//reset length -- highest non-zero limb, with sign of greater element
		u32 r = g_len;
		while ( r > 0 && v[r][i] == 0 ) { --r; }
		v[0][i] = r * sign(v, g);
		return r;
	}
}

/** 
 * Subtracts the j'th element of v from the i'th element of v.
 * Assumes there are enough limbs allocated in v to hold the result of the subtraction; 
 * max{size(v, i), size(v, j)} + 1 will do.
 * @return the number of limbs used by the i'th element of v after subtraction
 */
DEVICE_HOST static u32 sub(mpv v, u32 i, u32 j) {
	
	if ( same_sign(v, i, j) ) { //same sign, subtract
		
		u32 g, l;  //greater and lesser magnitude indices
		if ( cmpu(v, i, j) ) {
			g = i; l = j;
		} else {
			g = j; l = i;
		}
		u32 g_len = size(v, g), l_len = size(v, l);
		
		//subtract
		u32 b = sub_l(v, i, g, l, l_len);
		//propegate borrow (will not overflow, by construction)
		for (u32 k = l_len+1; k <= g_len; ++k) {
			v[k][i] = v[k][g] - b;
			b &= (v[k][i] == 0xFFFFFFFF);
		}
		
		//reset length -- highest non-zero limb, with sign of the i'th element or the negation 
		// of the sign of the j'th element, depending which was greater
		u32 r = g_len;
		while ( r > 0 && v[r][i] == 0 ) { --r; }
		v[0][i] = r * ( ((g == i) * sign(v, i)) + ((g == j) * -1 * sign(v, j)) );
		return r;
		
	} else {  //opposite signs, add
		
		u32 len = size(v, i), that_len = size(v, j);
	
		//ensure extra limbs are zeroed
		u32 k = len+1;
		for (; k <= that_len; ++k) { v[k][i] = 0; }
		
		//add
		u32 c = add_l(v, i, i, j, that_len);
		//propegate carry
		k = that_len+1;
		for (; k <= len; ++k) {
			v[k][i] += c;
			c &= (v[k][i] == 0);
		}
		v[k][i] = c;
		
		//reset length -- k, accounting for the possibility of no carry, with the previous sign
		u32 r = k - (c == 0);
		v[0][i] = neg_if(r, is_neg(v, i));
		return r;
		
	}
}

/** 
 * Multiplies the i'th element of v by the j'th element, storing the result in the r'th element.
 * Assumes there are enough limbs allocated in v to hold the result of the multiplication; 
 * size(v, i) + size(v, j) will do.
 * @return the number of limbs used by the r'th element of v after subtraction
 */
DEVICE_HOST static u32 mul(mpv v, u32 r, u32 i, u32 j) {
	u32 n = size(v, i), m = size(v, j);
	//special case zeros
	if ( n == 0 || m == 0 ) {
		v[0][r] = 0;
		return 0;
	}
	
	u32 l = n+m;
	
	//ensure limbs of r are zeroed
	for (u32 k = 1; k <= l; ++k) { v[k][r] = 0; }
	//multiply
	mul_l(v, r, i, j, n, m);
	
	//reset length -- l, accounting for possibility of short operands, with combination of signs
	l -= (v[l][r] == 0);
	v[0][r] = l;
	if ( ! same_sign(v, i, j) ) v[0][r] *= -1;
	return l;
}

/** 
 * Divides the i'th element of v by the j'th element, storing the result in the r'th element and 
 * the remainder in the i'th element.
 * Purposely throws a divide by zero error if the j'th element is zero. Assumes there are enough 
 * limbs allocated in v to perform the calculation; size(v, i) + 1 will do.
 * @return the number of limbs used by the r'th element of v after subtraction
 */
DEVICE_HOST static u32 div(mpv v, u32 r, u32 i, u32 j) {
	u32 n = size(v, i), m = size(v, j), q_l, r_l;
	
	//divide by zero
	if ( m == 0 ) return 1/m;
	
	//j'th element > i'th element
	if ( m > n ) {
		v[0][r] = 0;  //set quotient to zero
		              //all remainder, so remainder stays the same
		return 0;
	}
	
	if ( m == 1 ) {  //single-limb division
		v[1][i] = div1_l(v, r, i, v[1][j], n);
		
		r_l = (v[1][i] != 0);  //length of remainder
	} else {  //multi-limb division
		divn_l(v, r, i, j, n, m);
		
		for (r_l = m; r_l > 0 && v[r_l][i] == 0; --r_l) {}  //length of remainder
	}
	
	//reset quotient length
	q_l = (n-m) + (v[n-m+1][r] != 0);  //n-m, accounting for possibility of non-zero high limb
	v[0][r] = q_l;
	if ( ! same_sign(v, i, j) ) v[0][r] *= -1;  //with correct sign
	//reset remainder length
	v[0][i] = r_l * sign(v, i);  //remainder length, with correct sign
	
	return q_l;
}

namespace {
/** 
 * Converts half-byte values to corresponding hex characters.
 * @param i		half-byte value to convert
 * @return corresponding hexadecimal character (lowercase), or '_' if 
 * 		nonesuch.
 */
DEVICE_HOST char ch(int i) {
	switch ( i ) {
	case 0: return '0';
	case 1: return '1';
	case 2: return '2';
	case 3: return '3';
	case 4: return '4';
	case 5: return '5';
	case 6: return '6';
	case 7: return '7';
	case 8: return '8';
	case 9: return '9';
	case 10: return 'a';
	case 11: return 'b';
	case 12: return 'c';
	case 13: return 'd';
	case 14: return 'e';
	case 15: return 'f';
	default: return '_';
	}
}

/**
 * Converts characters to corresponding half-byte values.
 * @param c		character to convert
 * @return integer value corresponding to that hexadecimal character, 
 * 		or -1 if not a valid hex character
 */
DEVICE_HOST s32 unch(char c) {
	switch ( c ) {
		case '0': return 0;
		case '1': return 1;
		case '2': return 2;
		case '3': return 3;
		case '4': return 4;
		case '5': return 5;
		case '6': return 6;
		case '7': return 7;
		case '8': return 8;
		case '9': return 9;
		case 'A': case 'a': return 10;
		case 'B': case 'b': return 11;
		case 'C': case 'c': return 12;
		case 'D': case 'd': return 13;
		case 'E': case 'e': return 14;
		case 'F': case 'f': return 15;
		default: return -1;
	}
}

} /* unnamed */

/** 
 * Prints the integer into the given character buffer. 
 * Caller is responsible to ensure sufficient space in buffer; 
 * 8*size(v,i) + 2 will do.
 * @param c		The character buffer to print into. Will print 
 * 				value in hexadecimal (using uppercase letters), 
 * 				most-significant figure first, prefixed with '-' 
 * 				for negative values.
 */
DEVICE_HOST void print(const mpv v, u32 i, char* c) {
	//check for zero
	if ( is_zero(v, i) ) {
		c[0] = '0'; c[1] = '\0';
		return;
	}
	
	//check for negative
	if ( is_neg(v, i) ) {
		*c = '-'; ++c;
	}
	
	u32 j = size(v, i);
	
	//avoid leading zeros on most-significant limb
	limb x = v[j][i];
	if ( (x & 0xF0000000) != 0 ) goto f8;
	else if ( (x & 0x0F000000) != 0 ) goto f7;
	else if ( (x & 0x00F00000) != 0 ) goto f6;
	else if ( (x & 0x000F0000) != 0 ) goto f5;
	else if ( (x & 0x0000F000) != 0 ) goto f4;
	else if ( (x & 0x00000F00) != 0 ) goto f3;
	else if ( (x & 0x000000F0) != 0 ) goto f2;
	else goto f1;
	
	//convert limbs to characters
	while ( j > 0 ) {
		x = v[j][i];
		
		f8: *c = ch(x >> 28); ++c;
		f7: *c = ch((x >> 24) & 0xF); ++c;
		f6: *c = ch((x >> 20) & 0xF); ++c;
		f5: *c = ch((x >> 16) & 0xF); ++c;
		f4: *c = ch((x >> 12) & 0xF); ++c;
		f3: *c = ch((x >> 8) & 0xF); ++c;
		f2: *c = ch((x >> 4) & 0xF); ++c;
		f1: *c = ch(x & 0xF); ++c;
		
		--j;
	}
	
	//add trailing null
	*c = '\0';
}

/**
 * Loads a value from a string.
 * Caller is responsible to ensure sufficient limbs; 
 * (len+7)/8 will do.
 * @param v		The vector
 * @param i		The element
 * @param c		A string of hexadecimal digits, prefixed with '-' for a 
 * 				negative value
 * @param len	The length of c
 * @return the number of limbs used
 */
DEVICE_HOST u32 parse(mpv v, u32 i, const char* c, u32 len) {
	if ( len == 0 ) {
		v[0][i] = 0;
		return 0;
	}
	
	//check negative
	s32 used_l;  //limbs used
	if ( *c == '-' ) {
		used_l = -1; ++c; --len;
	} else {
		used_l = 1;
	}
	
	//trim leading zeros
	while ( len > 0 && *c == '0' ) { ++c; --len; }
	
	//check for zero value
	if ( len == 0 ) {
		v[0][i] = 0;
		return 0;
	}
	
	//calculate number of limbs (1/8th number of characters, round up)
	used_l *= ((len+7) >> 3);
	v[0][i] = used_l;
	
	//parse first character
	u32 j = abs(used_l);
	limb x = 0;
	switch ( len & 0x7 ) {
		case 0: x |= (unch(*c) << 28); ++c;
		case 7: x |= (unch(*c) << 24); ++c;
		case 6: x |= (unch(*c) << 20); ++c;
		case 5: x |= (unch(*c) << 16); ++c;
		case 4: x |= (unch(*c) << 12); ++c;
		case 3: x |= (unch(*c) <<  8); ++c;
		case 2: x |= (unch(*c) <<  4); ++c;
		case 1: x |= unch(*c);         ++c;
	}
	v[j][i] = x;
	--j;
	
	//parse remaining characters
	while ( j > 0 ) {
		x = 0;
		
		x |= (unch(*c) << 28); ++c;
		x |= (unch(*c) << 24); ++c;
		x |= (unch(*c) << 20); ++c;
		x |= (unch(*c) << 16); ++c;
		x |= (unch(*c) << 12); ++c;
		x |= (unch(*c) <<  8); ++c;
		x |= (unch(*c) <<  4); ++c;
		x |= unch(*c);         ++c;
		
		v[j][i] = x;
		--j;
	}
	
	return abs(used_l);
}
	
} /* namespace kilo */
