#pragma once

/** 
 * Main header for Kilo Multi-Precision Integer project.
 * Defines basic type (interleaved array of mp-ints), and operations on 
 * elements of that array.
 * 
 * By convention, arithmetic operations all take two index parameters, 
 * assigning the result to the first parameter and returning the number 
 * of limbs used by the new value; the caller is responsible for 
 * ensuring the vector is expanded to a sufficiently large number of 
 * limbs.
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
inline u32 size(const mpv v, int i) {
	return abs(static_cast<s32>(v[0][i]));
}

/** @return what is the sign of the i'th element of v */
inline s32 sign(const mpv v, int i) {
	s32 u = static_cast<s32>(v[0][i]);
	return ( u > 0 ) - ( u < 0 );
}

/** @return is the i'th element of v zero? */
inline bool is_zero(const mpv v, int i) {
	return v[0][i] == 0;
}

/** @return is the i'th element of v positive? */
inline bool is_pos(const mpv v, int i) {
	return static_cast<s32>(v[0][i]) > 0;
}

/** @return is the i'th element of v negative? */
inline bool is_neg(const mpv v, int i) {
	return static_cast<s32>(v[0][i]) < 0;
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
	v = w;
	return v;
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
 * @return the number of limbs in slot i
 */
DEVICE_HOST u32 assign(mpv v, u32 i, u32 j) {
	u32 nl = size(v, j);
	for (u32 k = 0; k <= nl; ++k) { v[k][i] = v[k][j]; }
	return nl;
}

/**
 * Swaps the values in two slots
 * @param v			The vector
 * @param i			One slot
 * @param j			The other slot
 * @return the maximum number of limbs used in either slot
 */
DEVICE_HOST u32 swap(mpv v, u32 i, u32 j) {
	u32 nl = size(v, i);
	u32 nj = size(v, j);
	if ( nj > nl ) { nl = nj; }
	
	limb t;
	for (u32 k = 0; k <= nl; ++k) { t = v[k][i]; v[k][i] = v[k][j]; v[k][j] = t; }
	return nl;
}

namespace {
/** 
 * Converts half-byte values to corresponding hex characters.
 * @param i		half-byte value to convert
 * @return corresponding hexadecimal character (uppercase), or '_' if 
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
	case 10: return 'A';
	case 11: return 'B';
	case 12: return 'C';
	case 13: return 'D';
	case 14: return 'E';
	case 15: return 'F';
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
DEVICE_HOST void print(const mpv v, int i, char* c) {
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
