/**
 * Simple host-side calculator to test kilomp library.
 * 
 * Command syntax:
 * cmd       := var '=' var        # assign variable to other variable
 *            | var '=' hexstring  # assign hex value to variable
 *            | var                # print variable
 *            | "swap" var var     # swap two variables
 *            | "neg" var          # negate a variable
 * var       := '$' [0-9]
 * hexstring := '-'? [0-9A-Fa-f]+
 * 
 * @author Aaron Moss
 */

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
}; /* struct mp_vars */

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
	kilo::u32 v1, v2;
	
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
	if ( s == std::string("=") ) {
		//handle assignment op
		if ( in.eof() ) {
			std::cerr << "Expected operand to '='" << std::endl;
			return;
		}
		
		in >> s;
		
		if ( ! in.eof() ) {
			std::cerr << "Too many arguments - expected nothing after `" << s << "'" << std::endl;
			return;
		}
		
		v2 = parse_var(s);
		if ( v2 == not_var ) {
			//assign constant
			vars.parse(v1, s);
		} else {
			//assign variable
			kilo::assign(vars.vs, v1, v2);
		}
		
		vars.print(v1);
		return;
	} else {
		std::cerr << "`" << s << "' is not an operator - expects '='" << std::endl;
		return;
	}
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
