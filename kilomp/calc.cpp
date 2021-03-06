/**
 * Simple host-side calculator to test kilomp library.
 * 
 * Command syntax (one command per line):
 * cmd       := var '=' var          # assign variable to other variable
 *            | var '=' hexstring    # assign hex value to variable
 *            | var                  # print variable
 *            | "swap" var var       # swap two variables
 *            | "neg" var            # negate a variable
 *            | "size" var           # print number of limbs in a variable
 *            | "cmp" var var        # compare two variables
 *            | var '+=' var         # add second variable to first
 *            | var '-=' var         # subtract second variable from first
 *            | var '=' var '*' var  # set first variable to the product of second and third
 *            | var '=' var '/' var  # set first variable to quotient of second and third
 *                                   # - remainder will be stored in second variable
 *            | "hex" var            # print values of underlying array for debugging
 * var       := '$' [0-9]
 * hexstring := '-'? [0-9A-Fa-f]+
 */

// Copyright 2013 Aaron Moss
//
// This file is part of KiloMP.
//
// KiloMP is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published 
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KiloMP is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KiloMP.  If not, see <https://www.gnu.org/licenses/>.

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

#include "kilomp.cuh"

struct mp_vars {
	kilo::mpv vs;  //variables
	kilo::u32 l;   //current # of limbs
	kilo::u32 n;   //# of variables
	
	mp_vars(kilo::u32 n_ = 10, kilo::u32 l_ = 4) : l(l_), n(n_) {
		vs = kilo::init_mpv(n, l);
	}
	
	~mp_vars() {
		kilo::clear(vs, l);
	}
	
	void ensure_limbs(kilo::u32 l_) {
		if ( l_ > l ) {
			vs = kilo::expand(vs, n, l, 2*l_);
			l = 2*l_;
		}
	}
	
	/** Parses s into the i'th element, ensuring that the element is large enough */
	void parse(kilo::u32 i, std::string s) {
		//ensure enough limbs
		int len = s.size();
		int limbs = (len+7)/8;
		ensure_limbs(limbs);
		
		//parse value
		kilo::parse(vs, i, s.c_str(), len);
	}
	
	/** Prints the i'th element to standard output. */
	void print(kilo::u32 i) {
		//allocate large enough buffer
		kilo::u32 len = kilo::size(vs, i);
		char buf[8*len+2];
		
		//print into buffer, then stdout
		kilo::print(vs,i,buf);
		std::cout << buf << std::endl;
	}
	
	/** Prints the hex values of the underlying array of the i'th element to standard output. */
	void hex(kilo::u32 i) {
		std::cout << "[";
		std::ios_base::fmtflags f = std::cout.flags();
		for (kilo::u32 j = 0; j <= l; ++j) {
			std::cout << " " << std::hex << std::setfill('0') << std::setw(8) << vs[j][i];
		}
		std::cout.flags(f);
		std::cout << " ]" << std::endl;
	}
}; /* struct mp_vars */

/** binary operator enum */
enum bin_op {
	asn,    /**< assignment operator */
	
	adda,   /**< addition operator */
	suba,   /**< subtraction operator */
	
	no_bin  /**< invalid operator */
}; /* enum bin_op */

/** ternary operator enum */
enum tri_op {
	mul,    /**< multiplication */
	dvn,	/**< division */
	
	no_tri  /**< invalid operator */
}; /* enum tri_op */

/** @return binary operator corresponding to s */
bin_op get_bin_op(const std::string& s) {
	using std::string;
	
	if ( s == string("=") ) return asn;
	
	else if ( s == string("+=") ) return adda;
	else if ( s == string("-=") ) return suba;
	
	else return no_bin;
}

/** @return ternary operator corresponding to s */
tri_op get_tri_op(const std::string& s) {
	using std::string;
	
	if ( s == string("*") ) return mul;
	else if ( s == string("/") ) return dvn;
	
	else return no_tri;
}

/** Failure flag for variable parsing */
kilo::u32 not_var = -1;

/** 
 * Parses a variable
 * @param s			The string to parse
 * @return the index of the variable, or `not_var` for failure
 */
kilo::u32 parse_var(const std::string& s) {
	if ( s[0] != '$' || s.size() != 2 )	return not_var;
	
	if ( s[1] >= '0' && s[1] <= '9' ) return s[1] - '0';
	else return not_var;
}

/** 
 * Parses a command line.
 * @param line		The line to parse
 * @param vars		The variable list
 */
void parse_cmd(std::string line, mp_vars& vars) {
	std::istringstream in(line);
	std::string s;
	kilo::u32 v1, v2, v3;
	
	if ( in.eof() ) return;  //ignore empty command
	
	in >> s;
	
	
	if ( s == std::string("swap") ) {  //handle swap command
		if ( in.eof() ) {
			std::cerr << "Expected arguments to `swap'" << std::endl;
			return;
		}
		
		in >> s;
		v1 = parse_var(s);
		if ( v1 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( in.eof() ) {
			std::cerr << "Expected second operand to `swap'" << std::endl;
			return;
		}
		
		in >> s;
		v2 = parse_var(s);
		if ( v2 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		kilo::swap(vars.vs, v1, v2);
		vars.print(v1);
		return;
		
	} else if ( s == std::string("neg") ) {  //handle negate command
		if ( in.eof() ) {
			std::cerr << "Expected arguments to `neg'" << std::endl;
			return;
		}
		
		in >> s;
		v1 = parse_var(s);
		if ( v1 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		kilo::neg(vars.vs, v1);
		vars.print(v1);
		return;
	} else if ( s == std::string("size") ) {  //handle size command
		if ( in.eof() ) {
			std::cerr << "Expected arguments to `size'" << std::endl;
			return;
		}
		
		in >> s;
		v1 = parse_var(s);
		if ( v1 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		std::cout << kilo::size(vars.vs, v1) << std::endl;
		return;
	} else if ( s == std::string("cmp") ) {  //handle compare command
		if ( in.eof() ) {
			std::cerr << "Expected arguments to `cmp'" << std::endl;
			return;
		}
		
		in >> s;
		v1 = parse_var(s);
		if ( v1 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( in.eof() ) {
			std::cerr << "Expected second operand to `cmp'" << std::endl;
			return;
		}
		
		in >> s;
		v2 = parse_var(s);
		if ( v2 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		std::cout << kilo::cmp(vars.vs, v1, v2) << std::endl;
		return;
	} else if ( s == std::string("hex") ) {  // handle hex command
		if ( in.eof() ) {
			std::cerr << "Expected arguments to `hex'" << std::endl;
			return;
		}
		
		in >> s;
		v1 = parse_var(s);
		if ( v1 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		vars.hex(v1);
		return;
	}
	
	//read variable
	v1 = parse_var(s);
	if ( v1 == not_var ) {
		std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
		return;
	}
	
	//handle print command
	if ( in.eof() ) {
		vars.print(v1);
		return;
	}
	
	in >> s;
	bin_op op = get_bin_op(s);
	
	//handle invalid operator
	if ( op == no_bin ) {
		std::cerr << "`" << s << "' is not an operator - expects `=', `+=', or `-='" << std::endl;
		return;
	}
	
	if ( in.eof() ) {
		std::cerr << "Expected operand to `" << s << "'" << std::endl;
		return;
	}
	
	if ( op == asn ) {  //handle assign operators
		in >> s;
		v2 = parse_var(s);
		
		if ( v2 == not_var ) {  //assign constant
			if ( ! in.eof() ) {
				std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
				return;
			}
			
			vars.parse(v1, s);
			vars.print(v1);
			return;
		}
		
		if ( in.eof() ) {  //assign variable
			kilo::assign(vars.vs, v1, v2);
			vars.print(v1);
			return;
		}
		
		//check for ternary operator
		in >> s;
		tri_op op2 = get_tri_op(s);
		if ( op2 == no_tri ) {
			std::cerr << "`" << s << "' is not an operator - expects `*' or '/'" << std::endl;
			return;
		}
		
		if ( in.eof() ) {
			std::cerr << "Expected operand to `" << s << "'" << std::endl;
			return;
		}
		
		in >> s;
		v3 = parse_var(s);
		if ( v3 == not_var ) {
			std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
			return;
		}
		
		switch( op2 ) {
		case mul: kilo::mul(vars.vs, v1, v2, v3); break;
		case dvn: kilo::div(vars.vs, v1, v2, v3); break;
		case no_tri: /* handled above -- ignore */ break;
		}
		
		vars.print(v1);
		return;
		
	}
	
	in >> s;
	
	if ( ! in.eof() ) {
		std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
		return;
	}
	
	v2 = parse_var(s);
	if ( v2 == not_var ) {
		std::cerr << "`" << s << "' is not a variable - expects '$' [0-9]" << std::endl;
		return;
	}
	
	switch( op ) {
	case adda: kilo::add(vars.vs, v1, v2); break;
	case suba: kilo::sub(vars.vs, v1, v2); break;
	case asn: case no_bin: /* handled above --ignore */ break;
	}
	
	vars.print(v1);
}

/** Parses input one line at a time until EOF */
int main(int argc, char** argv) {
	std::string line;
	mp_vars vars(10);  //array of 10 variables
	
	while ( std::cin.good() ) {
		std::getline(std::cin, line);
		if ( std::cin.eof() ) break;
		
		parse_cmd(line, vars);
	}
}
