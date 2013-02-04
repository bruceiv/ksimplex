#ifndef _CHIMP_CHIMP_CUH_
#define _CHIMP_CHIMP_CUH_

/** Main header for the "CUDA-Host Integrated Multi-Precision" (chimp) project. 
 *  Defines basic type, the chimpz (chimp integer).
 * 	
 *  @author Aaron Moss
 */

#include <climits>
#include <cstdlib>
#include <istream>
#include <string>
#include <ostream>

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

namespace chimp {
	
	/** CUDA-Host Integrated Multi-Precision Integer (chimpz) type.
	 *  NOTE: All code using chimpz will not run on compute capability below 
	 *  		2.0 (Fermi), and must be compiled with the -arch=sm_20 flag.
	 *  NOTE: chimpz assumes a 32-bit int, but works with either 32 or 64-bit 
	 *  		long. chimpz also assumes two's complement representation for 
	 *  		negative integers
	 *  chimpz has been optimized for CUDA in some places to avoid branches by 
	 *  using clever bit manipulations. In this, the author is indebted to Sean 
	 *  Anderson's site "Bit Twiddling Hacks" at 
	 *  http://graphics.stanford.edu/~seander/bithacks.html
	 */
	class chimpz {
	public:
		typedef u32 limb;
		static const int limb_size = 4;

	private:
		/** Internal-only constructor. Allocates a chimpz with the given values 
		 *  for alloc_l and used_l, and a zeroed limb array of size alloc_l.
		 *  @param a_l		value for alloc_l
		 *  @param u_l		value for used_l
		 */
		DEVICE_HOST chimpz(int a_l, int u_l) : alloc_l(a_l), used_l(u_l) {
			if ( alloc_l != 0 ) {
				l = (limb*)malloc(alloc_l*limb_size);
				for (int i = 0; i < alloc_l; ++i) l[i] = 0;
			} else {
				l = NULL;
			}
		}
		
		/** Expands the internal storage of this chimpz to hold a_l limbs
		 *  @param a_l		The new internal storage size. Must be larger than 
		 *  				the current storage size. Should be a power of two.
		 */
		DEVICE_HOST void expand(int a_l) {
			//allocate new memory
			alloc_l = a_l;
			limb* l2 = (limb*)malloc(alloc_l*limb_size);
			
			//copy over values from old memory
			int i;
			for (i = 0; i < abs(used_l); ++i) l2[i] = l[i];
			for (; i < alloc_l; ++i) l2[i] = 0;
			
			//free old memory and replace with new
			if ( l != NULL ) free(l);
			l = l2;
		}
		
		/** Converts half-byte values to corresponding hex characters.
		 *  @param i		half-byte value to convert
		 *  @return corresponding hexadecimal character (uppercase), or '_' if 
		 *  		nonesuch.
		 */
		DEVICE_HOST static char ch(int i) {
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
		
		/** Converts characters to corresponding half-byte values.
		 *  @param c		character to convert
		 *  @return integer value corresponding to that hexadecimal character, 
		 *  		or -1 if not a valid hex character
		 */
		DEVICE_HOST static s32 unch(char c) {
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
		
		/** Prints a limb array, starting at index i and working down to zero
		 *  @param c		The character string to print into
		 *  @param a		The limb array to print
		 *  @param i		The index to start printing at
		 *  @return pointer to the end of the printed string
		 */
		DEVICE_HOST static char* print(char* c, const limb* l, int i) {
			//avoid leading zeros on most-significant limb
			limb x = l[i];
			if ( (x & 0xF0000000) != 0 ) goto f8;
			else if ( (x & 0x0F000000) != 0 ) goto f7;
			else if ( (x & 0x00F00000) != 0 ) goto f6;
			else if ( (x & 0x000F0000) != 0 ) goto f5;
			else if ( (x & 0x0000F000) != 0 ) goto f4;
			else if ( (x & 0x00000F00) != 0 ) goto f3;
			else if ( (x & 0x000000F0) != 0 ) goto f2;
			else goto f1;
			
			//convert limbs to characters
			while ( i >= 0 ) {
				x = l[i];
				
				f8: *c = ch(x >> 28); ++c;
				f7: *c = ch((x >> 24) & 0xF); ++c;
				f6: *c = ch((x >> 20) & 0xF); ++c;
				f5: *c = ch((x >> 16) & 0xF); ++c;
				f4: *c = ch((x >> 12) & 0xF); ++c;
				f3: *c = ch((x >> 8) & 0xF); ++c;
				f2: *c = ch((x >> 4) & 0xF); ++c;
				f1: *c = ch(x & 0xF); ++c;
				
				--i;
			}
			
			//add trailing null
			*c = '\0';
			
			return c;
		}
		
		/** Loads the values in the chimpz from the given character array. 
		 *  Meant for initialization, thus the limb array should be unallocated 
		 *  when called, or the memory will leak.
		 *  @param c		a string of hexadecimal digits, optionally prefixed 
		 *  				with '-' for a negative number
		 *  @param len		the length of the character string
		 */
		DEVICE_HOST void parse(const char* c, int len) {
			if ( len == 0 ) {
				//set up null value
				alloc_l = 0;
				used_l = 0;
				l = NULL;
			} else {
				if ( *c == '-' ) {
					used_l = -1; ++c; --len;
				} else {
					used_l = 1;
				}
				
				//trim leading zeros
				while ( len > 0 && *c == '0' ) { ++c; --len; }
				
				//check for zero value
				if ( len == 0 ) {
					alloc_l = 0;
					used_l = 0;
					l = NULL;
					return;
				}
				
				//calculate number of limbs used (1/8th of the number of 
				// characters, round up)
				used_l *= ((len+7) >> 3);
				
				//calculate the first power of two greater than twice used_l
				// to allocate memory for
				unsigned int v = abs(used_l);
				unsigned int r; unsigned int s;
				r = ( v > 0xFFFF ) << 4; v >>= r;
				s = ( v >   0xFF ) << 3; v >>= s; r |= s;
				s = ( v >    0xF ) << 2; v >>= s; r |= s;
				s = ( v >    0x3 ) << 1; v >>= s; r |= s;
				                                  r |= (v >> 1);
				alloc_l = 1 << (r + 1);
				
				//allocate memory
				l = (limb*)malloc(alloc_l*limb_size);
				
				//parse characters
				int i = abs(used_l)-1;
				int j = len & 0x7;
				while ( i >= 0 ) {
					limb x = 0;
					switch ( j ) {
						case 0: x |= (unch(*c) << 28); ++c;
						case 7: x |= (unch(*c) << 24); ++c;
						case 6: x |= (unch(*c) << 20); ++c;
						case 5: x |= (unch(*c) << 16); ++c;
						case 4: x |= (unch(*c) << 12); ++c;
						case 3: x |= (unch(*c) <<  8); ++c;
						case 2: x |= (unch(*c) <<  4); ++c;
						case 1: x |= unch(*c);         ++c;
					}
					j = 0;
					
					l[i] = x;
					--i;
				}
			}
		}
		
		/** Equality function.
		 *  @param that		chimpz to compare
		 *  @return true for equal, false for unequal
		 */
		DEVICE_HOST bool eq(const chimpz& that) const {
			if ( used_l != that.used_l ) return false;
			
			for (int i = 0; i < abs(used_l); ++i) {
				if ( l[i] != that.l[i] ) return false;
			}
			
			return true;
		}
		
		/** Comparison function.
		 *  @param that		chimpz to compare
		 *  @return 0 for equal, 1 for greater than that, -1 for less than that
		 */
		DEVICE_HOST int cmp(const chimpz& that) const {
			if ( used_l > that.used_l ) {
				return 1;
			} else if ( used_l < that.used_l ) {
				return -1;
			}
			
			//compare magnitudes - the values have the same sign, and are 
			// within a single limb's value range of each other (or the above 
			// would have eliminated them), but we need to account properly 
			// for the sign (greater magnitude negative is a less-than, greater 
			// magnitude positive is a greater-than)
			for (int i = abs(used_l)-1; i >= 0; --i) {
				if ( l[i] == that.l[i] ) {
					continue;
				} else if ( l[i] > that.l[i] ) {
					return (used_l > 0) - (used_l < 0); //sgn(this)
				} else /* if ( l[i] < that.l[i] ) */ {
					return (used_l < 0) - (used_l > 0); //-sgn(this)
				}
			}
			
			return 0;
		}
		
		/** Unsigned (magnitude) comparison function.
		 *  @param a		first chimpz to compare
		 *  @param b		second chimpz to compare
		 *  @return true for |a| >= |b|, false for |a| < |b|
		 */
		DEVICE_HOST static bool cmpu(const chimpz& a, const chimpz& b) {
			int a_len = abs(a.used_l), b_len = abs(b.used_l);
			if ( a_len > b_len ) {
				return true;
			} else if ( a_len < b_len ) {
				return false;
			}
			
			//compare values
			for (int i = a_len-1; i >= 0; --i) {
				if ( a.l[i] == b.l[i] ) {
					continue;
				} else if ( a.l[i] > b.l[i] ) {
					return true;
				} else /* if ( a.l[i] < b.l[i] ) */ {
					return false;
				}
			}
			
			return true;
		}
		
		/** Addition function - performs z = x + y.
		 *  @param z		The output array (should have at least n limbs 
		 *  				allocated) (may be either of x or y, but if not all 
		 *  				limbs should be zeroed)
		 *  @param x		The first operand
		 *  @param y		The second operand
		 *  @param n		Number of limbs to add
		 *  @return the carry bit from the last addition
		 */
		DEVICE_HOST static int add(limb* z, const limb* x, const limb* y, 
								   int n) {
			int c = 0;
			for (int i = 0; i < n; ++i) {
				int t = x[i] + y[i];
				z[i] = t + c;
				c = (t < x[i]);
			}
			return c;
		}
		
		/** Subtraction function - performs z = x - y.
		 *  @param z		The output array (should have at least n limbs 
		 *  				allocated) (may alias x, but if not all limbs 
		 *  				should be zeroed)
		 *  @param x		The first operand
		 *  @param y		The second operand
		 *  @param n		Number of limbs to subtract
		 *  @return the borrow bit from the last subtraction
		 */
		DEVICE_HOST static int sub(limb* z, const limb* x, const limb* y, 
								   int n) {
			int b = 0;
			for (int i = 0; i < n; ++i) {
				int t = x[i] - y[i];
				z[i] = t - b;
				b = (t > x[i]);
			}
			return b;
		}
		
		/** Multiplication function - performs z = x*y.
		 *  @param z		The output array (should have at least n_x + n_y 
		 *  				limbs allocated, initially zeroed) (must not alias 
		 *  				x or y)
		 *  @param x		The first operand
		 *  @param y		The second operand
		 *  @param n_x		The number of limbs in x
		 *  @param n_y		The number of limbs in y
		 */
		DEVICE_HOST static void mul(limb* z, const limb* x, const limb* y, 
								   int n_x, int n_y) {
			for (int i = 0; i < n_y; ++i) {
				u32 c = 0;
				for (int j = 0; j < n_x; ++j) {
					u64 x_j = x[j], y_i = y[i], z_ij = z[i+j];
					
					//We know uv won't overflow as (taking n = 2^32), 
					// (n-1)^2 + 2(n-1) = n^2 - 1 (that is, multiply any two 
					// limbs, and add any other two limbs, and the result can 
					// be no greater than two limbs)
					z_ij += c;
					u64 uv = (x_j * y_i) + z_ij;
					
					//put low order bits of multiply into output
					z[i+j] = (u32)(uv & 0xFFFFFFFF);
					//keep high order bits in carry
					c = (u32)(uv >> 32);
				}
				z[i+n_x] = c;
			}
		}
		
		/** Division function - performs z = x/y (truncating toward zero).
		 *  @param z		The output array (should have at least n_x limbs 
		 *  				allocated, initially zeroed) (may alias x)
		 *  @param x		The first operand
		 *  @param y		The limb to divide through
		 *  @param n_x		The number of limbs in x (should be >= 1)
		 *  @return the remainder of the division
		 */
		DEVICE_HOST static limb div1(limb* z, const limb* x, limb y, int n_x) {
			u32 c = 0;
			u64 t;
			for (int i = n_x-1; i >= 0; --i) {
				t = c; t <<= 32; t |= x[i];
				z[i] = (t / y);
				c = (t % y);
			}
			return c;
		}
		
		/** Division function - performs z = x/y (truncating toward zero).
		 *  @param z		The output array (should have at least n_x - n_y 
		 *  				limbs allocated, initially zeroed) (must not alias 
		 *  				x or y)
		 *  @param x		The first operand, also the output remainder 
		 *  				(will not be preserved by function, should be 
		 *  				copied beforehand if desired preserved) (should 
		 *  				have at least n_x + 1 limbs allocated)
		 *  @param y		The second operand (is modified during the method, 
		 *  				but restored at the end, so can be safely 
		 *  				const-casted in most cases)
		 *  @param n_x		The number of limbs in x (should be >= n_y)
		 *  @param n_y		The number of limbs in y (should be >= 2 - use div1 
		 *  				otherwise)
		 */
		DEVICE_HOST static void divn(limb* z, limb* x, limb* y, 
									 int n_x, int n_y) {
			int n_d = n_x - n_y;
			
			//normalize so that y[n_y-1] >= 2^31
			//equivalently, shift y left such that there is a 1 its high bit
			
			//d will contain the high-order bit index (from 1) of y[n_y-1]. 
			// 31-d will be the normalization shift
			u32 v = y[n_y-1];
			u32 d = (v > 0xFFFF) << 4; v >>= d;
			u32 s = (v > 0xFF  ) << 3; v >>= s; d |= s;
			    s = (v > 0xF   ) << 2; v >>= s; d |= s;
				s = (v > 0x3   ) << 1; v >>= s; d |= s;
				                                d |= (v >> 1);
			d = 31 - d;
			
			//normalize x and y by the given amount
			if ( d > 0 ) {
				lsh(x, x, d, n_x);
				lsh(y, y, d, n_y);
			}
			
			//main loop (divides first n_y digits of the quotient by y)
			for (int j = n_d; j >= 0; --j) {
				//determine trial divisor qh
				u32 qh, rh;
				u64 t1, t2, t3;
				if ( x[j+n_y] == y[n_y-1] ) {
					qh = 0xFFFFFFFF;
				} else {
					t1 = x[j+n_y]; t1 <<= 32; t1 |= x[j+n_y-1];
					qh = t1 / y[n_y-1];
					rh = t1 % y[n_y-1];
					
					t2 = y[n_y-2]; t2 *= qh;
					t3 = rh; t3 <<= 32; t3 |= x[j+n_y-2];
					while ( t2 > t3 ) {
						//update trial divisior if needed
						--qh;
						if ( rh + y[n_y-1] < rh ) break;
						rh += y[n_y-1];
						
						t2 = y[n_y-2]; t2 *= qh;
						t3 = rh; t3 <<= 32; t3 |= x[j+n_y-2];
					}
				}
				
				//replace x by (x - qh*y)
				u32 b = 0, m = 0;
				for (int i = 0; i < n_y; ++i) {
					t1 = qh; t1 *= y[i]; t1 += b;
					m = (t1 & 0xFFFFFFFF);
					b = (t1 >> 32);
					
					if ( x[j+i] < m ) {
						++b;
					}
					
					x[j+i] -= m;
				}
				
				if ( b > x[j+n_y] ) {
					//check case where qh was 1 too big (if so, add a y back in)
					x[j+n_y] -= b;
					qh--;
					add(x+j, x+j, y, n_y);
					//ignore final carry, it cancels the borrow from earlier
				} else {
					x[j+n_y] -= b;
				}
				
				z[j] = qh;
			}
				
			//Unnormalize x and y
			if ( d > 0 ) {
				rsh(x, x, d, n_x+1);
				rsh(y, y, d, n_y);
			}
		}
		
		/** Left-shift function - performs z = x << s
		 *  @param z		The output limb array (should have at least 
		 *  				n+(s/32)+1 limbs allocated, may alias x)
		 *  @param x		The input limb array
		 *  @param s		The number of bits to shift left by
		 *  @param n		The number of limbs in x
		 */
		DEVICE_HOST static void lsh(limb* z, const limb* x, int s, int n) {
			int s_l = s >> 5 /* s / 32 */; //number of limbs to shift
			int s_b = s & 0x1F /* s % 32 */; //number of bits to shift
			
			int mask = ((1 << s_b) - 1) << (32 - s_b); //mask for high bits
			int u_s = 32 - s_b;
			
			//get high order bits of high limb
			z[n+s_l] = (x[n-1] & mask) >> u_s;
			//shift remaining limbs
			for (int i = n-1; i > 0; --i) {
				// get low order bits of current limb, and then high-order of 
				// next back
				z[i+s_l] = (x[i] << s_b) | ((x[i-1] & mask) >> u_s);
			}
			//get low order bits of low limb
			z[s_l] = x[0] << s_b;
			//zero low-order limbs
			for (int i = s_l - 1; i >= 0; --i) {
				z[i] = 0;
			}
		}
		
		/** Right-shift function - performs z = x >> s
		 *  @param z		The ouptut limb array (should have at least 
		 *  				n limbs allocated, may alias x)
		 *  @param x		The input limb array
		 *  @param s		The number of bits to shift right by
		 *  @param n		The number of limbs in x (should be at least s/32)
		 */
		DEVICE_HOST static void rsh(limb* z, const limb* x, int s, int n) {
			int s_l = s >> 5 /* s/32 */; //number of limbs to shift
			int s_b = s & 0x1F /* s%32 */; //number of bits to shift
			
			int mask = (1 << s_b) - 1;
			int u_s = 32 - s_b;
			
			//right shift limbs
			for (int i = 0; i < n-1-s_l; ++i) {
				z[i] = ((x[i+1+s_l] & mask) << u_s) | (x[i+s_l] >> s_b);
			}
			//get low order bits of high limb
			z[n-1-s_l] = x[n-1] >> s_b;
			//zero high-order limbs
			for (int i = n-s_l; i < n; ++i) {
				z[i] = 0;
			}
		}
		
	public:
		/** Default constructor - value at zero. Does not allocate any 
		 *  storage.
		 */
		DEVICE_HOST chimpz() : alloc_l(0), used_l(0) { l = NULL; }
		
		/** Int constructor - value at x
		 *  @param x		The value to load the chimpz with
		 */
		DEVICE_HOST chimpz(s32 x) : alloc_l(2) {
			l = (limb*)malloc(2*limb_size);
			
			if ( x == 0 ) {
				used_l = 0;
				l[0] = 0; l[1] = 0;
			} else if ( x == 0x80000000 /* -2^32 */ ) {
				used_l = -1;
				l[0] = 0x80000000; l[1] = 0;
			} else if ( x < 0 ) {
				used_l = -1;
				l[0] = abs(x); l[1] = 0;
			} else /* if ( x > 0 ) */ {
				used_l = 1;
				l[0] = x; l[1] = 0;
			}
		}
		
		/** Unsigned int constructor - value at x
		 *  @param x		The value to load the chimpz with
		 */
		DEVICE_HOST chimpz(u32 x) : alloc_l(2) {
			l = (limb*)malloc(2*limb_size);
			
			if ( x == 0 ) {
				used_l = 0;
				l[0] = 0; l[1] = 0;
			} else /* if ( x > 0 ) */ {
				used_l = 1;
				l[0] = x; l[1] = 0;
			}
		}
		
		/** Long int constructor - value at x
		 *  @param x		The value to load the chimpz with
		 */
		DEVICE_HOST chimpz(s64 x) : alloc_l(4) {
			l = (limb*)malloc(4*limb_size);
			
			if ( x == 0 ) {
				used_l = 0;
				l[0] = 0; l[1] = 0;
			} else if ( x == 0x8000000000000000 /* -2^64 */ ) {
				used_l = -2;
				l[0] = 0; l[1] = 0x80000000; l[2] = 0; l[3] = 0;
			} else if ( x < 0 ) {
				x = abs(x);
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				l[2] = 0; l[3] = 0;
				used_l = -1 - ( l[1] != 0 );
			} else /* if ( x > 0 ) */ {
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				l[2] = 0; l[3] = 0;
				used_l = 1 + ( l[1] != 0 );
			}
		}
		
		/** Unsigned long int constructor - value at x
		 *  @param x		The value to load the chimpz with
		 */
		DEVICE_HOST chimpz(u64 x) : alloc_l(4) {
			l = (limb*)malloc(4*limb_size);
			
			if ( x == 0 ) {
				used_l = 0;
				l[0] = 0; l[1] = 0; l[2] = 0; l[3] = 0;
			} else /* if ( x > 0 ) */ {
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				l[2] = 0; l[3] = 0;
				used_l = 1 + ( l[1] != 0 );
			}
		}
		
		/** C-string constructor. 
		 *  @param c		A null-terminated string containing a hexadecimal 
		 *  				number (using uppercase letters) representing the 
		 *  				magnitude of the number, optionally prefixed by '-' 
		 *  				for a negative number.
		 */
		DEVICE_HOST chimpz(const char* c) {
			//get length of array
			int len = 0; const char *cc = c;
			while ( *cc != '\0' ) { ++len; ++cc; }
			
			parse(c, len);
		}
		
		/** STL-string constructor.
		 *  @param s		A string containing a hexadecimal number (using 
		 *  				uppercase letters) representing the magnitude of 
		 *  				the number, optionally prefixed by '-' for a 
		 *  				negative number.
		 */
		HOST_ONLY chimpz(std::string s) {
			parse(s.c_str(), s.size());
		}
		
		/** Copy constructor
		 *  @param that		The chimpz to copy
		 */
		DEVICE_HOST chimpz(const chimpz& that) 
				: alloc_l(that.alloc_l), used_l(that.used_l) {
			if ( alloc_l != 0 ) {
				l = (limb*)malloc(alloc_l*limb_size);
				for (int i = 0; i < alloc_l; ++i) { l[i] = that.l[i]; }
			} else {
				l = NULL;
			}
		}
		
		/** Raw constructor
		 *  @param a_l		alloc_l value
		 *  @param u_l		used_l value
		 *  @param l_p		limb array (will be copied by this constructor)
		 */
		DEVICE_HOST chimpz(int a_l, int u_l, limb* l_p) 
				: alloc_l(a_l), used_l(u_l) {
			if ( l_p == NULL ) {
				l = NULL;
			} else {
				l = (limb*)malloc(a_l*limb_size);
				for (int i = 0; i < a_l; ++i) { l[i] = l_p[i]; }
			}
		}
		
		/** Destructor */
		DEVICE_HOST ~chimpz() { if ( l != NULL ) free(l); }
		
		/** Signed int assignment operator.
		 *  @param x		The integer to assign
		 *  @return this, modified to equal x. The limb array will be expanded 
		 *  		as needed, but will not be contracted
		 */
		DEVICE_HOST chimpz& operator= (s32 x) {
			if ( alloc_l < 2 ) expand(2);
			
			int i;
			if ( x == 0 ) {
				used_l = 0;
				i = 0;
			} else if ( x == 0x80000000 /* -2^32 */ ) {
				used_l = -1;
				l[0] = 0x80000000;
				i = 1;
			} else if ( x < 0 ) {
				used_l = -1;
				l[0] = abs(x);
				i = 1;
			} else /* if ( x > 0 ) */ {
				used_l = 1;
				l[0] = x;
				i = 1;
			}
			
			//zero the rest
			while ( i < alloc_l ) { l[i++] = 0; }
			
			return *this;
		}
		
		/** Unsigned int assignment operator.
		 *  @param x		The integer to assign
		 *  @return this, modified to equal x. The limb array will be expanded 
		 *  		as needed, but will not be contracted
		 */
		DEVICE_HOST chimpz& operator= (u32 x) {
			if ( alloc_l < 2 ) expand(2);
			
			int i;
			if ( x == 0 ) {
				used_l = 0;
				i = 0;
			} else /* if ( x > 0 ) */ {
				used_l = 1;
				l[0] = x;
				i = 1;
			}
			
			//zero the rest
			while ( i < alloc_l ) { l[i++] = 0; }
			
			return *this;
		}
		
		/** Signed long assignment operator.
		 *  @param x		The integer to assign
		 *  @return this, modified to equal x. The limb array will be expanded 
		 *  		as needed, but will not be contracted
		 */
		DEVICE_HOST chimpz& operator= (s64 x) {
			if ( alloc_l < 4 ) expand(4);
			
			if ( x == 0 ) {
				used_l = 0;
			} else if ( x == 0x8000000000000000 /* -2^64 */ ) {
				used_l = -2;
				l[0] = 0; l[1] = 0x80000000;
			} else if ( x < 0 ) {
				x = abs(x);
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				used_l = -1 - ( l[1] != 0 );
			} else /* if ( x > 0 ) */ {
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				used_l = 1 + ( l[1] != 0 );
			}
			
			//zero the rest
			int i = abs(used_l);
			while ( i < alloc_l ) { l[i++] = 0; }
			
			return *this;
		}
		
		/** Unsigned long assignment operator.
		 *  @param x		The integer to assign
		 *  @return this, modified to equal x. The limb array will be expanded 
		 *  		as needed, but will not be contracted
		 */
		DEVICE_HOST chimpz& operator= (u64 x) {
			if ( alloc_l < 4 ) expand(4);
			
			if ( x == 0 ) {
				used_l = 0;
				l[0] = 0; l[1] = 0; l[2] = 0; l[3] = 0;
			} else /* if ( x > 0 ) */ {
				l[0] = x & 0xFFFFFFFF; 
				l[1] = (x & 0xFFFFFFFF00000000) >> 32;
				l[2] = 0; l[3] = 0;
				used_l = 1 + ( l[1] != 0 );
			}
			
			//zero the rest
			int i = abs(used_l);
			while ( i < alloc_l ) { l[i++] = 0; }
			
			return *this;
		}
		
		/** Assignment operator.
		 *  @param that		The chimpz to copy
		 *  @return this, modified to be a copy of that. The limb array will be 
		 *  		expanded as needed, but will not be contracted unless that 
		 *  		is an empty chimpz, in which case it will be nulled
		 */
		DEVICE_HOST chimpz& operator= (const chimpz& that) {
			if ( that.alloc_l > alloc_l ) {
				//that has a bigger allocated array, make this match
				alloc_l = that.alloc_l;
				
				if ( l != NULL ) free(l);
				l = (limb*)malloc(alloc_l*limb_size);
			} else if ( that.alloc_l == 0 ) {
				//that has no allocated array, make this match (not strictly 
				// needed, but provides a nice way to clear a memory allocation)
				alloc_l = that.alloc_l;
				
				if ( l != NULL ) free(l);
				l = NULL;
			}
			
			used_l = that.used_l;
			
			for (int i = 0; i < alloc_l; ++i) l[i] = that.l[i];
			return *this;
		}
		
		/** Equality operator.
		 *  @param that		The chimpz to test
		 *  @return true for equal, false for unequal
		 */
		DEVICE_HOST bool operator== (const chimpz& that) const {
			return eq(that);
		}
		
		/** Inequality operator.
		 *  @param that		The chimpz to test
		 *  @return true for unequal, false for equal
		 */
		DEVICE_HOST bool operator!= (const chimpz& that) const {
			return ! eq(that);
		}
		
		/** Less-than operator.
		 *  @param that		The chimpz to test
		 *  @return true for less than that, false otherwise
		 */
		DEVICE_HOST bool operator< (const chimpz& that) const {
			return cmp(that) == -1;
		}
		
		/** Less-than-or-equal operator.
		 *  @param that		The chimpz to test
		 *  @return true for less than or equal that, false otherwise
		 */
		DEVICE_HOST bool operator<= (const chimpz& that) const {
			return cmp(that) != 1;
		}
		
		/** Greater-than operator.
		 *  @param that		The chimpz to test
		 *  @return true for greater than that, false otherwise
		 */
		DEVICE_HOST bool operator> (const chimpz& that) const {
			return cmp(that) == 1;
		}
		
		/** Greater-than-or-equal operator.
		 *  @param that		The chimpz to test
		 *  @return true for greater than or equal that, false otherwise
		 */
		DEVICE_HOST bool operator>= (const chimpz& that) const {
			return cmp(that) != -1;
		}
		
		/** Unary negation operator.
		 *  @return the negation of this chimpz
		 */
		DEVICE_HOST chimpz operator- () const {
			chimpz neg(*this);
			neg.used_l *= -1;
			return neg;
		}
		
		/** Addition operator.
		 *  @return this + that
		 */
		DEVICE_HOST chimpz operator+ (const chimpz& that) const {
			if ( used_l == 0 ) return chimpz(that); //this is zero
			if ( that.used_l == 0 ) return chimpz(*this); //that is zero
			
			int len = abs(used_l), that_len = abs(that.used_l);
			int a_l, u_l, n, n_l;
			limb* l_l;
			
			if ( (used_l ^ that.used_l) >= 0 ) {
				//same sign: add
				
				if ( len >= that_len ) {
					//this is longer, use its parameters (doubling alloc_l if 
					// this is full)
					u_l = used_l; 
					a_l = alloc_l << (len == alloc_l);
					n = that_len;
					n_l = len;
					l_l = l;
				} else {
					//that is longer, use its parameters (doubling alloc_l if 
					// that is full)
					u_l = that.used_l;
					a_l = that.alloc_l << (that_len == that.alloc_l);
					n = len;
					n_l = that_len;
					l_l = that.l;
				}
				
				//allocate a new chimpz
				chimpz ret(a_l, u_l);
				
				//add the shared limbs
				int c = add(ret.l, l, that.l, n);
				//add the unshared limbs (propegating carry)
				for (int i = n; i < n_l; ++i) {
					ret.l[i] = l_l[i] + c;
					c = (ret.l[i] == 0);
				}
				ret.l[n_l] = c;
				//account for possible change in used_l from carry (+0 if carry 
				// 0, +1 if carry 1 and ret > 0, -1 if carry 1 and ret < 0)
				ret.used_l += ( c == 1 ) * ( ( u_l > 0 ) - ( u_l < 0 ) );
				
				return ret;
			} else {
				//opposite signs: subtract
				
				limb* l_s;
				
				if ( cmpu(*this, that) ) {
					//this is larger, use its parameters (doubling alloc_l if  
					// this is full)
					u_l = used_l; 
					a_l = alloc_l << (len == alloc_l);
					n = that_len;
					n_l = len;
					l_l = l;
					l_s = that.l;
					
				} else {
					//that is longer, use its parameters (doubling alloc_l if 
					// that is full)
					u_l = that.used_l;
					a_l = that.alloc_l << (that_len == that.alloc_l);
					n = len;
					n_l = that_len;
					l_l = that.l;
					l_s = l;
				}
				
				//allocate a new chimpz
				chimpz ret(a_l, u_l);
				
				//subtract the shared limbs
				int b = sub(ret.l, l_l, l_s, n);
				//propegate the borrow through the unshared limbs
				for (int i = n; i < n_l; ++i) {
					ret.l[i] = l_l[i] - b;
					b = (ret.l[i] == 0xFFFFFFFF);
				}
				//account for possible change in used_l from borrow (+0 if last 
				// element != 0, -1 * sign(ret) otherwise)
				ret.used_l += 
					( ret.l[n_l-1] == 0 ) * -1 * ( ( u_l > 0 ) - ( u_l < 0 ) );
				
				return ret;
			}
		}
		
		/** Assignment-addition operator.
		 *  @return this += that
		 */
		DEVICE_HOST chimpz& operator+= (const chimpz& that) {
			if ( used_l == 0 ) return (*this = that);
			if ( that.used_l == 0 ) return *this;
			
			int len = abs(used_l), that_len = abs(that.used_l);
			int u_l, n, n_l;
			limb* l_l;
			
			//expand limb array as needed (doubling size if full)
			if ( alloc_l < that.alloc_l ) {
				expand(that.alloc_l << ( that_len == that.alloc_l ));
			} else if ( len == alloc_l ) {
				expand(alloc_l << 1);
			}
			
			if ( (used_l ^ that.used_l) >= 0 ) {
				//same sign: add
				
				if ( len >= that_len ) {
					u_l = used_l;
					n = that_len;
					n_l = len;
					l_l = l;
				} else {
					u_l = that.used_l;
					n = len;
					n_l = that_len;
					l_l = that.l;
				}
				
				//add the shared limbs
				int c = add(l, l, that.l, n);
				//add the unshared limbs (propegating carry)
				for (int i = n; i < n_l; ++i) {
					l[i] = l_l[i] + c;
					c = (l[i] == 0);
				}
				l[n_l] = c;
				
				//account for possible change in used_l from carry (+0 if carry 
				// 0, +1 if carry 1 and ret > 0, -1 if carry 1 and ret < 0)
				used_l = u_l + (( c == 1 ) * ( ( u_l > 0 ) - ( u_l < 0 ) ));
			} else {
				//opposite signs, subtract
				limb* l_s;
				
				if ( cmpu(*this, that) ) {
					u_l = used_l;
					n = that_len;
					n_l = len;
					l_l = l;
					l_s = that.l;
				} else {
					u_l = that.used_l;
					n = len;
					n_l = that_len;
					l_l = that.l;
					l_s = l;
				}
				
				//subtract the shared limbs
				int b = sub(l, l_l, l_s, n);
				//propegate the borrow through the unshared limbs
				for (int i = n; i < n_l; ++i) {
					l[i] = l_l[i] - b;
					b = (l[i] == 0xFFFFFFFF);
				}
				//account for possible change in used_l from borrow (+0 if last 
				// element != 0, -1 * sign(ret) otherwise)
				used_l = u_l + 
					( l[n_l-1] == 0 ) * -1 * ( ( u_l > 0 ) - ( u_l < 0 ) );
			}
			
			return *this;
		}
		
		/** Subtraction operator.
		 *  @return this - that
		 */
		DEVICE_HOST chimpz operator- (const chimpz& that) const {
			if ( used_l == 0 ) return -that; //this is zero
			if ( that.used_l == 0 ) return chimpz(*this); //that is zero
			
			int len = abs(used_l), that_len = abs(that.used_l);
			int a_l, u_l, n, n_l;
			limb* l_l;
			
			if ( (used_l ^ that.used_l) < 0 ) {
				//opposite signs: add
				
				if ( len >= that_len ) {
					//this is longer, use its parameters (doubling alloc_l if 
					// this is full)
					u_l = used_l; 
					a_l = alloc_l << (len == alloc_l);
					n = that_len;
					n_l = len;
					l_l = l;
				} else {
					//that is longer, use its parameters (doubling alloc_l if 
					// that is full)
					u_l = -that.used_l;
					a_l = that.alloc_l << (that_len == that.alloc_l);
					n = len;
					n_l = that_len;
					l_l = that.l;
				}
				
				//allocate a new chimpz
				chimpz ret(a_l, u_l);
				
				//add the shared limbs
				int c = add(ret.l, l, that.l, n);
				//add the unshared limbs (propegating carry)
				for (int i = n; i < n_l; ++i) {
					ret.l[i] = l_l[i] + c;
					c = (ret.l[i] == 0);
				}
				ret.l[n_l] = c;
				//account for possible change in used_l from carry (+0 if carry 
				// 0, +1 if carry 1 and ret > 0, -1 if carry 1 and ret < 0)
				ret.used_l += ( c == 1 ) * ( ( u_l > 0 ) - ( u_l < 0 ) );
				
				return ret;
			} else {
				//same sign: subtract
				
				limb* l_s;
				
				if ( cmpu(*this, that) ) {
					//this is larger, use its parameters (doubling alloc_l if  
					// this is full)
					u_l = used_l; 
					a_l = alloc_l << (len == alloc_l);
					n = that_len;
					n_l = len;
					l_l = l;
					l_s = that.l;
					
				} else {
					//that is larger, use its parameters (doubling alloc_l if 
					// that is full)
					u_l = -that.used_l;
					a_l = that.alloc_l << (that_len == that.alloc_l);
					n = len;
					n_l = that_len;
					l_l = that.l;
					l_s = l;
				}
				
				//allocate a new chimpz
				chimpz ret(a_l, u_l);
				
				//subtract the shared limbs
				int b = sub(ret.l, l_l, l_s, n);
				//propegate the borrow through the unshared limbs
				for (int i = n; i < n_l; ++i) {
					ret.l[i] = l_l[i] - b;
					b = (ret.l[i] == 0xFFFFFFFF);
				}
				//account for possible change in used_l from borrow (+0 if last 
				// element != 0, -1 * sign(ret) otherwise)
				ret.used_l += 
					( ret.l[n_l-1] == 0 ) * ( ( u_l < 0 ) - ( u_l > 0 ) );
				
				return ret;
			}
		}
		
		/** Assignment-subtraction operator.
		 *  @return this -= that
		 */
		DEVICE_HOST chimpz& operator-= (const chimpz& that) {
			if ( used_l == 0 ) return (*this = -that);
			if ( that.used_l == 0 ) return *this;
			
			int len = abs(used_l), that_len = abs(that.used_l);
			int u_l, n, n_l;
			limb* l_l;
			
			//expand limb array as needed (doubling size if full)
			if ( alloc_l < that.alloc_l ) {
				expand(that.alloc_l << ( that_len == that.alloc_l ));
			} else if ( len == alloc_l ) {
				expand(alloc_l << 1);
			}
			
			if ( (used_l ^ that.used_l) < 0 ) {
				//opposite signs: add
				
				if ( len >= that_len ) {
					u_l = used_l;
					n = that_len;
					n_l = len;
					l_l = l;
				} else {
					u_l = -that.used_l;
					n = len;
					n_l = that_len;
					l_l = that.l;
				}
				
				//add the shared limbs
				int c = add(l, l, that.l, n);
				//add the unshared limbs (propegating carry)
				for (int i = n; i < n_l; ++i) {
					l[i] = l_l[i] + c;
					c = (l[i] == 0);
				}
				l[n_l] = c;
				
				//account for possible change in used_l from carry (+0 if carry 
				// 0, +1 if carry 1 and ret > 0, -1 if carry 1 and ret < 0)
				used_l = u_l + (( c == 1 ) * ( ( u_l > 0 ) - ( u_l < 0 ) ));
			} else {
				//same sign, subtract
				limb* l_s;
				
				if ( cmpu(*this, that) ) {
					u_l = used_l;
					n = that_len;
					n_l = len;
					l_l = l;
					l_s = that.l;
					
				} else {
					u_l = -that.used_l;
					n = len;
					n_l = that_len;
					l_l = that.l;
					l_s = l;
				}
				
				//subtract the shared limbs
				int b = sub(l, l_l, l_s, n);
				//propegate the borrow through the unshared limbs
				for (int i = n; i < n_l; ++i) {
					l[i] = l_l[i] - b;
					b = (l[i] == 0xFFFFFFFF);
				}
				//account for possible change in used_l from borrow (+0 if last 
				// element != 0, -1 * sign(ret) otherwise)
				used_l = u_l + 
					( l[n_l-1] == 0 ) * ( ( u_l < 0 ) - ( u_l > 0 ) );
			}
			
			return *this;
		}
		
		/** Multiplication operator.
		 *  @return this * that
		 */
		DEVICE_HOST chimpz operator* (const chimpz& that) const {
			//account for multiply by zero
			if ( used_l == 0 || that.used_l == 0 ) return chimpz(alloc_l, 0);
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//calculate limbs used as well as an allocation big enough to hold 
			// them
			int r_len = len + that_len;
			int a_l = alloc_l;
			while ( r_len >= a_l ) a_l <<= 1;
			//calculate sign of returned integer
			int u_l = r_len;
			if ( ( used_l ^ that.used_l ) < 0 ) u_l *= -1;
			
			//create new int and multiply
			chimpz ret(a_l, u_l);
			mul(ret.l, l, that.l, len, that_len);
			
			//account for possible change in used_l from small operands (+0 if 
			// last element != 0, -1 * sgn(ret) otherwise)
			ret.used_l += 
				( ret.l[r_len-1] == 0 ) * ( ( u_l < 0 ) - ( u_l > 0 ) );
			
			return ret;
		}
		
		/** Assignment-multiplication operator.
		 *  @return this *= that
		 */
		DEVICE_HOST chimpz& operator*= (const chimpz& that) {
			//account for multiply by zero
			if ( used_l == 0 ) return *this; 
			if ( that.used_l == 0 ) return (*this = that);
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//calculate limbs used as well as an allocation big enough to hold 
			// them
			int r_len = len + that_len;
			int a_l = alloc_l;
			while ( r_len >= a_l ) a_l <<= 1;
			//calculate new sign
			int u_l = r_len;
			if ( ( used_l ^ that.used_l ) < 0 ) u_l *= -1;
			
			//create new limb array and multiply
			limb* r_l = (limb*)malloc(a_l*limb_size);
			for (int i = 0; i < a_l; ++i) r_l[i] = 0;
			mul(r_l, l, that.l, len, that_len);
			
			//put new parameters into this
			alloc_l = a_l;
			//account for possible change in used_l from small operands (+0 if 
			// last element != 0, -1 * sgn(ret) otherwise)
			used_l = u_l +
				( r_l[r_len-1] == 0 ) * ( ( u_l < 0 ) - ( u_l > 0 ) );
			if ( l != NULL ) free(l);
			l = r_l;
			
			return *this;
		}
		
		/** Division operator.
		 *  @param that		Divisor - if 0, will purposely divide by zero to 
		 *  				throw error
		 *  @return this / that (truncating)
		 */
		DEVICE_HOST chimpz operator/ (const chimpz& that) const {
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//divide by zero
			if ( that_len == 0 ) return chimpz(1/that.used_l);
			
			chimpz quot(alloc_l, 0);
			
			//other one bigger
			if ( len < that_len ) {
				quot.used_l = 0;
				return quot;
			}
			
			if ( that_len == 1 ) {
				//perform division by single limb
				div1(quot.l, l, that.l[0], len);
				//adjust used_l
				quot.used_l = (len - ( quot.l[len-1] == 0 ));
				if ( (used_l ^ that.used_l) < 0 ) quot.used_l *= -1;
				return quot;
			}
			
			int a_l = alloc_l << ( alloc_l == len );
			chimpz rem(a_l, 0); rem = *this;
			
			//perform division
			divn(quot.l, rem.l, that.l, len, that_len);
			
			//correct used_l
			for (int i = len-that_len; i >= 0; --i) {
				if ( quot.l[i] != 0 ) { quot.used_l = i+1; break; }
			}
			if ( ( used_l ^ that.used_l ) < 0 ) quot.used_l *= -1;
			
			return quot;
		}
		
		/** Assignment-division operator.
		 *  @return this /= that (truncating)
		 */
		DEVICE_HOST chimpz& operator/= (const chimpz& that) {
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//divide by zero
			if ( that_len == 0 ) { used_l /= that.used_l; return *this; }
			
			//other one bigger
			if ( len < that_len ) {
				for (int i = 0; i < used_l; ++i) l[i] = 0;
				used_l = 0;
				return *this;
			}
			
			if ( that_len == 1 ) {
				//perform division by single limb
				div1(l, l, that.l[0], len);
				//adjust used_l one toward zero if its high-order limb is zeroed
				used_l -= ( l[len-1] == 0 ) * ( (used_l > 0) - (used_l < 0) );
				if ( (used_l ^ that.used_l) < 0 ) used_l *= -1;
				return *this;
			}
			
			int a_l = alloc_l << ( alloc_l == len );
			chimpz rem(a_l, 0); rem = *this;
			
			//perform division
			divn(l, rem.l, that.l, len, that_len);
			
			//correct used_l
			int old_u_l = used_l;
			for (int i = len-that_len; i >= 0; --i) {
				if ( l[i] != 0 ) { used_l = i+1; break; }
			}
			if ( ( old_u_l ^ that.used_l ) < 0 ) used_l *= -1;
			
			return *this;
		}
		
		/** Bitwise and operator.
		 *  @return this & that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Return value 
		 *  		takes sign of this)
		 */
		DEVICE_HOST chimpz operator& (const chimpz& that) const {
			//account for zero
			if ( used_l == 0 || that.used_l == 0 ) { chimpz z; return z; }
			
			chimpz ret(alloc_l, 0);
			
			int len = abs(used_l), that_len = abs(that.used_l);
			//r_len = min(len, that_len)
			int r_len = len ^ ((len ^ that_len) & -(len < that_len));
			
			int i;
			for (i = 0; i < r_len; ++i) ret.l[i] = l[i] & that.l[i];
			for (; i < alloc_l; ++i) ret.l[i] = 0;
			for (i = r_len-1; i >= 0; --i) {
				if ( ret.l[i] != 0 ) {
					ret.used_l = (i+1) * ( ( used_l > 0 ) - ( used_l < 0 ) );
					break;
				}
			}
			
			return ret;
		}
		
		/** Assignment-bitwise and operator.
		 *  @return this &= that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Sign will not 
		 *  		change)
		 */
		DEVICE_HOST chimpz& operator&= (const chimpz& that) {
			//account for zero
			if ( used_l == 0 ) return *this;
			if ( that.used_l == 0 ) return (*this = 0);
			
			int len = abs(used_l), that_len = abs(that.used_l);
			//r_len = min(len, that_len)
			int r_len = len ^ ((len ^ that_len) & -(len < that_len));
			
			int i;
			for (i = 0; i < r_len; ++i) l[i] &= that.l[i];
			for (; i < alloc_l; ++i) l[i] = 0;
			int old_u_l = used_l;
			used_l = 0;
			for (i = r_len-1; i >= 0; --i) {
				if ( l[i] != 0 ) {
					used_l = (i+1) * ( ( old_u_l > 0 ) - ( old_u_l < 0 ) );
					break;
				}
			}
			
			return *this;
		}
		
		/** Bitwise or operator.
		 *  @return this | that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Return value 
		 *  		takes sign of this)
		 */
		DEVICE_HOST HOST_ONLY chimpz operator| (const chimpz& that) const {
			//account for zero
			if ( used_l == 0 ) { return chimpz(that); }
			if ( that.used_l == 0 ) { return chimpz(*this); }
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			int a_l, s_len, r_len;
			limb* r_l;
			if ( len >= that_len ) {
				s_len = that_len;
				r_len = len;
				a_l = alloc_l;
				r_l = l;
			} else {
				s_len = len;
				r_len = that_len;
				a_l = that.alloc_l;
				r_l = that.l;
			}
			
			chimpz ret(a_l, 0);
			
			int i;
			for (i = 0; i < s_len; ++i) ret.l[i] = l[i] | that.l[i];
			for (; i < a_l; ++i) ret.l[i] = r_l[i];
			ret.used_l = r_len * ( ( used_l > 0 ) - ( used_l < 0 ) );
			
			return ret;
		}
		
		/** Assignment-bitwise or operator.
		 *  @return this |= that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Sign will not 
		 *  		change)
		 */
		DEVICE_HOST chimpz& operator|= (const chimpz& that) {
			//account for zero
			if ( used_l == 0 ) { return (*this = that); }
			if ( that.used_l == 0 ) { return *this; }
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//ensure enough space
			if ( alloc_l < that.alloc_l ) expand(that.alloc_l);
			
			int i;
			for (i = 0; i < that_len; ++i) l[i] |= that.l[i];
			if ( len < that_len ) {
				used_l = that_len * ( ( used_l > 0 ) - ( used_l < 0 ) );
			}
			
			return *this;
		}
		
		/** Bitwise xor operator.
		 *  @return this ^ that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Return value 
		 *  		takes sign of this)
		 */
		DEVICE_HOST chimpz operator^ (const chimpz& that) const {
			//account for zero
			if ( used_l == 0 ) { return chimpz(that); }
			if ( that.used_l == 0 ) { return chimpz(*this); }
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			int a_l, s_len, r_len;
			limb* r_l;
			if ( len >= that_len ) {
				s_len = that_len;
				r_len = len;
				a_l = alloc_l;
				r_l = l;
			} else {
				s_len = len;
				r_len = that_len;
				a_l = that.alloc_l;
				r_l = that.l;
			}
			
			chimpz ret(a_l, 0);
			
			int i;
			for (i = 0; i < s_len; ++i) ret.l[i] = l[i] ^ that.l[i];
			for (; i < a_l; ++i) ret.l[i] = r_l[i];
			for (i = r_len-1; i >= 0; --i) {
				if ( ret.l[i] != 0 ) {
					ret.used_l = (i+1) * ( ( used_l > 0 ) - ( used_l < 0 ) );
					break;
				}
			}
			
			return ret;
		}
		
		/** Assignment-bitwise xor operator.
		 *  @return this ^= that (where one chimpz is shorter than the other, 
		 *  		the extra length is assumed to contain zeros. Sign will not 
		 *  		change)
		 */
		DEVICE_HOST chimpz& operator^= (const chimpz& that) {
			//account for zero
			if ( used_l == 0 ) { return (*this = that); }
			if ( that.used_l == 0 ) { return *this; }
			
			int len = abs(used_l), that_len = abs(that.used_l);
			
			//ensure enough space
			if ( alloc_l < that.alloc_l ) expand(that.alloc_l);
			
			int i;
			for (i = 0; i < that_len; ++i) l[i] ^= that.l[i];
			if ( len < that_len ) {
				used_l = that_len * ( ( used_l > 0 ) - ( used_l < 0 ) );
			} else if ( len == that_len ) {
				int old_u_l = used_l;
				used_l = 0;
				for (i = that_len-1; i >= 0; --i) {
					if ( l[i] != 0 ) {
						used_l = (i+1) * ( ( old_u_l > 0 ) - ( old_u_l < 0 ) );
						break;
					}
				}
			}
			
			return *this;
		}
		
		/** Left bitshift operator.
		 *  @param s		The number of bits to shift leftward
		 *  @return a copy of this chimpz shifted leftward by that many bits. 
		 *  		The sign will be the same as this chimpz
		 */
		DEVICE_HOST chimpz operator<< (unsigned int s) const {
			//account for zero and zero shift
			if ( used_l == 0 || s == 0 ) { return chimpz(*this); }
			
			int len = abs(used_l);
			//calculate current limbs used plus those used by new bits
			int r_len = len + (s>>5) + 1;
			//calculate allocation big enough for them
			int a_l = alloc_l;
			while ( r_len >= a_l ) a_l <<= 1;
			//multiply sign back in for return
			int u_l = r_len * ((used_l > 0) - (used_l < 0));
			
			chimpz ret(a_l, u_l);
			
			//perform left shift
			lsh(ret.l, l, s, len);
			
			//adjust for lack of high bits in high limb (if high limb is zero, 
			// will decrement a positive used_l or increment a negative used_l)
			ret.used_l -= ( ret.l[r_len-1] == 0 ) * ((u_l > 0) - (u_l < 0));
			
			return ret;
		}
		
		/** Assignment-left bitshift operator.
		 *  @param s		The number of bits to shift leftward
		 *  @return this chimpz, shifted leftward by that many bits. The sign 
		 *  		will be unchanged
		 */
		DEVICE_HOST chimpz operator<<= (unsigned int s) {
			//account for zero and zero shift
			if ( used_l == 0 || s == 0 ) { return *this; }
			
			int len = abs(used_l);
			//calculate current limbs used plus those used by new bits
			int r_len = len + (s>>5) + 1;
			//calculate allocation big enough for them
			int a_l = alloc_l;
			while ( r_len >= a_l ) a_l <<= 1;
			//multiply sign back in for new value
			used_l = r_len * ((used_l > 0) - (used_l < 0));
			
			//expand storage if needed
			if ( a_l < alloc_l ) expand(a_l);
			
			//perform left shift
			lsh(l, l, s, len);
			
			//adjust for lack of high bits in high limb (if high limb is zero, 
			// will decrement a positive used_l or increment a negative used_l)
			used_l -= ( l[r_len-1] == 0 ) * ((used_l > 0) - (used_l < 0));
			
			return *this;
		}
		
		/** Right bitshift operator.
		 *  @param s		The number of bits to shift rightward
		 *  @return a copy of this chimpz shifted rightward by that many bits. 
		 *  		The sign will be the same as this chimpz
		 */
		DEVICE_HOST chimpz operator>> (unsigned int s) const {
			//account for zero and zero shift
			if ( used_l == 0 || s == 0 ) { return chimpz(*this); }
			
			int len = abs(used_l);
			//calculate current limbs used plus those used by new bits
			int r_len = len - (s>>5);
			//multiply sign back in for return
			int u_l = r_len * ((used_l > 0) - (used_l < 0));
			
			chimpz ret(alloc_l, u_l);
			
			//perform right shift
			rsh(ret.l, l, s, len);
			
			//adjust for lack of high bits in high limb (if high limb is zero, 
			// will decrement a positive used_l or increment a negative used_l)
			ret.used_l -= ( ret.l[r_len-1] == 0 ) * ((u_l > 0) - (u_l < 0));
			
			return ret;
		}
		
		/** Assignment-right bitshift operator.
		 *  @param s		The number of bits to shift rightward
		 *  @return this chimpz, shifted rightward by that many bits. The sign 
		 *  		will be unchanged
		 */
		DEVICE_HOST chimpz operator>>= (unsigned int s) {
			//account for zero and zero shift
			if ( used_l == 0 || s == 0 ) { return *this; }
			
			int len = abs(used_l);
			//calculate current limbs used plus those used by new bits
			int r_len = len - (s>>5);
			//multiply sign back in for new value
			used_l = r_len * ((used_l > 0) - (used_l < 0));
			
			//perform right shift
			rsh(l, l, s, len);
			
			//adjust for lack of high bits in high limb (if high limb is zero, 
			// will decrement a positive used_l or increment a negative used_l)
			used_l -= ( l[r_len-1] == 0 ) * ((used_l > 0) - (used_l < 0));
			
			return *this;
		}
		
		/** Computes the greatest common denominator (gcd) of u and v
		 *  @param u		The first parameter
		 *  @param v		The second parameter
		 *  @return gcd(u, v) (0 for both u, v zero)
		 */
		DEVICE_HOST static chimpz gcd(chimpz u, chimpz v) {
			if ( u.used_l == 0 ) return chimpz(v);
			if ( v.used_l == 0 ) return chimpz(u);
			
			if ( u.used_l < 0 ) u.used_l = -u.used_l;
			if ( v.used_l < 0 ) v.used_l = -v.used_l;
			
			//factor out all shared multiples of two
			unsigned int shift = 0;
			int i;
			for (i = 0; (u.l[i] | v.l[i]) == 0; ++i) {
				shift += 32;
			}
			u32 x = u.l[i] | v.l[i];
			if ( (x & 0x1) == 0 ) {
				shift += 1;
				if ( (x & 0xFFFF) == 0 ) {
					x >>= 16;
					shift += 16;
				}
				if ( (x & 0xFF) == 0 ) {
					x >>= 8;
					shift += 8;
				}
				if ( (x & 0xF) == 0 ) {
					x >>= 4;
					shift += 4;
				}
				if ( (x & 0x3) == 0 ) {
					x >>= 2;
					shift += 2;
				}
				shift -= (x & 0x1);
			}
			u >>= shift;
			v >>= shift;
			
			//TODO finish
			//factor all remaining multiples of two out of u
			int s = 0;
			for (i = 0; u.l[i] == 0; ++i) {
				s += 32;
			}
			x = u.l[i];
			if ( (x & 0x1) == 0 ) {
				s += 1;
				if ( (x & 0xFFFF) == 0 ) {
					x >>= 16;
					s += 16;
				}
				if ( (x & 0xFF) == 0 ) {
					x >>= 8;
					s += 8;
				}
				if ( (x & 0xF) == 0 ) {
					x >>= 4;
					s += 4;
				}
				if ( (x & 0x3) == 0 ) {
					x >>= 2;
					s += 2;
				}
				s -= (x & 0x1);
			}
			u >>= s;
			
			//here forward, u always odd
			do {
				s = 0;
				for (i = 0; v.l[i] == 0; ++i) {
					s += 32;
				}
				x = v.l[i];
				if ( (x & 0x1) == 0 ) {
					s += 1;
					if ( (x & 0xFFFF) == 0 ) {
						x >>= 16;
						s += 16;
					}
					if ( (x & 0xFF) == 0 ) {
						x >>= 8;
						s += 8;
					}
					if ( (x & 0xF) == 0 ) {
						x >>= 4;
						s += 4;
					}
					if ( (x & 0x3) == 0 ) {
						x >>= 2;
						s += 2;
					}
					s -= (x & 0x1);
				}
				v >>= s;
				
				if ( ! cmpu(v, u) ) {
					swap(u, v);
				}
				
				v -= u;
			} while ( v.used_l != 0 );
			
			return u << shift;
		}
		
		/** Size of the integer
		 *  @return # of limbs used
		 */
		DEVICE_HOST int size() const { 
			return abs(used_l);
		}
		
		/** Sign of the integer
		 *  @return 0 for zero, 1 for positive, -1 for negative
		 */
		DEVICE_HOST int sign() const {
			return ( used_l > 0 ) - ( used_l < 0 );
		}
		
		/** Signed limbs used.
		 *  @return the number of limbs used, times the sign of the integer
		 */
		DEVICE_HOST int used() const { return used_l; }
		
		/** Allocated memory. 
		 *  @return the number of allocated limbs
		 */
		DEVICE_HOST int limbs() const { return alloc_l; }
		
		
		DEVICE_HOST const limb* data() const { return l; }
		
		/** Swaps contents of u, v
		 *  @param u		The first chimpz to swap
		 *  @param v		The second chimpz to swap
		 */
		DEVICE_HOST static void swap(chimpz& u, chimpz& v) {
			int t = u.alloc_l; u.alloc_l = v.alloc_l; v.alloc_l = t;
			t = u.used_l; u.used_l = v.used_l; v.used_l = t;
			limb* p = u.l; u.l = v.l; v.l = p;
		}
		
		/** Prints the integer into the given character buffer (caller is 
		 *  responsible to ensure sufficient space in buffer).
		 *  @param c		The character buffer to print into. Will print 
		 *  				value in hexadecimal (using uppercase letters), 
		 *  				most-significant figure first, prefixed with '-' 
		 *  				for negative values.
		 */
		DEVICE_HOST void print(char* c) const {
			if ( alloc_l == 0 || used_l == 0 ) {
				c[0] = '0'; c[1] = '\0';
			} else {
				int i;
				
				if ( used_l < 0 ) {
					i = abs(used_l) - 1;
					*c = '-'; ++c;
				} else {
					i = used_l - 1;
				}
				
				print(c, l, i);
			}
		}
		
		/** Casts to unsigned int. Returns the least significant bits of this, 
		 *  ignoring sign (GMP functions the same way)
		 */
		DEVICE_HOST operator unsigned int () const {
			return ( alloc_l == 0 ) ? 0 : l[0];
		}
		
		/** Casts to string. Prints into temporary character buffer. */
		HOST_ONLY operator std::string () const {
			char c[abs(used_l)*4+2];
			print(c);
			std::string s = std::string(c);
			return s;
		}
		
	private:
		int alloc_l;	/**< allocated limbs */
		int used_l;		/**< magnitude: used limbs; sign: integer sign */
		limb *l;		/**< limb array (unsigned, least-significant limb 
						     first) */
	}; /* class chimpz */
	
	/** Output operator */
	HOST_ONLY std::ostream& operator<< (std::ostream& out, const chimpz& z) {
		out << std::string(z);
		return out;
	}
	
	/** Input operator */
	HOST_ONLY std::istream& operator>> (std::istream& in, chimpz& z) {
		std::string s;
		in >> s;
		z = chimpz(s);
		return in;
	}
	
} /* namespace chimp */

#endif /* _CHIMP_CHIMP_CUH_ */
