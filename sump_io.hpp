#ifndef _SUMP_SUMP_IO_HPP_
#define _SUMP_SUMP_IO_HPP_
/** I/O methods for the "Simplex Using Multi-Precision" (sump) project
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <istream>
#include <ostream>
#include <sstream>
#include <string>

#include "sump.hpp"

namespace sump {
	
	/** Reads an index_list from the given input stream. The returned list 
	 *  should be freed by the caller.
	 *  
	 *  @param in		The input stream to read from
	 *  @param n		The length of the index_list (returned list will be 
	 *  				one longer, with 0 in the 0th index)
	 *  @return the index_list read
	 */
	index_list readIndexList(std::istream& in, ind n) {
		index_list l = allocIndexList(n);
		for (ind i = 1; i <= n; ++i) in >> l[i];
		return l;
	}
	
	/** Writes an index_list to the given output stream.
	 *  
	 *  @param out		The output stream to write to
	 *  @param l		The list to write
	 *  @param n		The length of the list (not accounting for the empty 
	 *  				0th index)
	 */
	void writeIndexList(std::ostream& out, const index_list l, ind n) {
		if ( n == 0 ) return;
		out << l[1];
		for (ind i = 2; i <= n; ++i) out << " " << l[i];
	}
	
	/** I/O methods for numeric elements.
	 *  
	 *  @param T		The type of the element to read
	 *  @param Ty		The number type of the element to read
	 */
	template<typename T, element_type Ty>
	struct element_io_traits {
		
		/** Input method.
		 *  
		 *  @param in		The input stream to read from
		 *  @return The value read
		 */
		static T read(std::istream& in) {
			T x;
			in >> x;
			return x;
		}
		
		/** Output method.
		 *  
		 *  @param out		The output stream to write to
		 *  @param x		The value to write
		 */
		static void write(std::ostream& out, const T& x) {
			out << x;
		}
	};
	
	/** Input method for floating-point. */
	template<typename T>
	struct element_io_traits< T, floating_point > {
		
		static T read(std::istream& in) {
			std::string s;
			std::stringstream ss;
			in >> s;
			
			// Search for fraction divider
			std::size_t i = s.find('/');
			
			if ( i == std::string::npos ) {
				// No fraction divider
				ss.str(s);
				T f;
				ss >> f;
				return f;
			} else {
				//manually divide fraction
				ss.str(s.substr(0, i));
				T n;
				ss >> n;
				
				ss.clear();
				ss.str(s.substr(i+1));
				T d;
				ss >> d;
				return n/d;
			}
		}
		
		static void write(std::ostream& out, const T& f) {
			out << f;
		}
	};
	
	/** Input method for integers. */
	template<typename T>
	struct element_io_traits< T, integral > {
		
		static T read(std::istream& in) {
			std::string s;
			std::stringstream ss;
			in >> s;
			
			//NOTE: this implementation ignores any fractional or 
			// floating-point portion of the number, but you can't put that in 
			// an integer anyway...
			ss.str(s);
			T z;
			ss >> z;
			
			return z;
		}
		
		static void write(std::ostream& out, const T& z) {
			out << z;
		}
	};
	
	/** Reads a single element.
	 *  
	 *  @param T		The type of element to read
	 *  @param in		The stream to read from
	 *  @return the element read
	 */
	template<typename T>
	T readElement(std::istream& in) {
		return element_io_traits< T, element_traits< T >::elType >::read(in);
	}
	
	/** Writes a single element.
	 *  
	 *  @param T		The type of element to write
	 *  @param out		The stream to write to
	 *  @param x		The element to write
	 */
	template<typename T>
	void writeElement(std::ostream& out, const T& x) {
		element_io_traits< T, element_traits< T >::elType >::write(out, x);
	}
	
	/** Reads a matrix in from the given input stream. The returned matrix 
	 *  should be freed by the caller.
	 *  
	 *  @param T		The element type
	 *  @param in		The input stream to read from
	 *  @param n		The maximum valid row index (from 0)
	 *  @param d		The maximum valid column index (from 0)
	 *  @return the matrix read
	 */
	template<typename T>
	T* readMatrix(std::istream& in, ind n, ind d) {
		T* m = allocMat< T >(n, d);
		for (ind i = 0; i <= n; ++i) for (ind j = 0; j <= d; ++j) {
			m[i*(d+1)+j] = readElement< T >(in);
		}
		
		return m;
	}
	
	/** Writes a matrix to a given output stream.
	 * 
	 *  @param T		The element type
	 *  @param out		The output stream
	 *  @param m		The matrix
	 *  @param n		The maximum valid row index (from 0)
	 *  @param d		The maximum valid column index (from 0)
	 */
	template<typename T>
	void writeMatrix(std::ostream& out, const T* m, ind n, ind d) {
		for (ind i = 0; i <= n; ++i) {
			writeElement< T >(out, m[i*(d+1)]);
			for (ind j = 1; j <= d; ++j) {
				out << " ";
				writeElement< T >(out, m[i*(d+1)+j]);
			}
			out << std::endl;
		}
	}
	
	/** Prints a tableau to the given output stream.
	 *  
	 *  @param Tab		The type of the tableau
	 *  @param out		The output stream
	 *  @param t		The tableau
	 */
	template<typename Tab>
	void printTableau(std::ostream& out, const Tab& t) {
		typedef typename Tab::value_type element;
		typedef typename Tab::matrix_type matrix;
		
		// Write size and dimension
		ind n = t.size();
		ind d = t.dim();
		out << n << " " << d << std::endl;
		
		if ( n == 0 || d == 0 ) return;
		
		// Write basis and cobasis
		const index_list b = t.basis();
		writeIndexList(out, b, n);
		out << " : ";
		
		const index_list c = t.cobasis();
		writeIndexList(out, c, d);
		out << std::endl;
		
		// Write matrix
		const matrix& m = t.mat();
		for (ind i = 0; i <= n; ++i) {
			out << "[" << b[i] << "]";
			out << "[" << c[0] << "]";
			writeElement< element >(out, m[i*(d+1)]);
			for (ind j = 1; j <= d; ++j) {
				out << " [" << c[j] << "]";
				writeElement< element >(out, m[i*(d+1)+j]);
			}
			out << std::endl;
		}
	}
}
#endif /* SUMP_SUMP_IO_HPP_ */
