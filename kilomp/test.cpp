/**
 * Test harness for kilomp library.
 * Takes a list of expressions (one per line) on standard input, and compares kilomp to gmp.
 *
 * Expression syntax:
 * <op> <value> <value>
 *
 * <op> is one of '+', '-', '*', '/' - '#' can also be used to indicate a skipped comment line
 * <value> should be a signed hexadecimal string (e.g. "-e4" or "6b")
 *
 * @author Aaron Moss
 */

#include <iostream>
#include <string>
#include <sstream>

#include <gmp.h>

#include "kilomp.cuh"

/** Encapsulates comparison between kilomp and gmp */
struct arith_test {
private:
	// Disable copying
	arith_test(const arith_test& that);
	arith_test& operator = (const arith_test& that);
	
	/** Reads operands into array indices 1 and 2 */
	void read_ops(const std::string& a, const std::string& b) {
		// ensures sufficient limbs in kilomp vector
		kilo::u32 len = ( a.size() >= b.size() ) ? a.size() : b.size();
		kilo::u32 limbs = (len+7)/8;
		if ( limbs > l ) {
			kops = kilo::expand(kops, 3, l, 2*limbs);
			l = 2*limbs;
		}
		
		// reads kilomp values
		kilo::parse(kops, 1, a.c_str(), a.size());
		kilo::parse(kops, 2, b.c_str(), b.size());
		
		// reads the gmp values
		mpz_set_str(gops[1], a.c_str(), 16);
		mpz_set_str(gops[2], b.c_str(), 16);
	}
	
public:	
	arith_test() : l(4) {
		kops = kilo::init_mpv(3, l);
		for (int i = 0; i < 3; ++i) { mpz_init(gops[i]); }
	}
	
	~arith_test() {
		kilo::clear(kops, l);
		for (int i = 0; i < 3; ++i) { mpz_clear(gops[i]); }
	}
	
	/** Add a and b */
	void add(const std::string& a, const std::string& b) {
		// Read popoer
		read_ops(a, b);
		
		// Add kilomp
		kilo::assign(kops, 0, 1);
		kilo::add(kops, 0, 2);
		
		// Add gmp
		mpz_add(gops[0], gops[1], gops[2]);
	}
	
	/** Subtract b from a */
	void sub(const std::string& a, const std::string& b) {
		// Read popoer
		read_ops(a, b);
		
		// Subtract kilomp
		kilo::assign(kops, 0, 1);
		kilo::sub(kops, 0, 2);
		
		// Subtract gmp
		mpz_sub(gops[0], gops[1], gops[2]);
	}
	
	/** Multiply a and b */
	void mul(const std::string& a, const std::string& b) {
		// Read popoer
		read_ops(a, b);
		
		// Multiply kilomp
		kilo::mul(kops, 0, 1, 2);
		
		// Multiply gmp
		mpz_mul(gops[0], gops[1], gops[2]);
	}
	
	/** Divide a by b - should be exact */
	void div(const std::string& a, const std::string& b) {
		// Read popoer
		read_ops(a, b);
		
		// Multiply kilomp
		kilo::div(kops, 0, 1, 2);
		
		// Multiply gmp
		mpz_divexact(gops[0], gops[1], gops[2]);
	}
	
	/** @return kilomp result */
	std::string get_kilomp() {
		// Allocate buffer
		kilo::u32 len = kilo::size(kops, 0);
		char buf[8*len+2];
		
		// Print and return string
		kilo::print(kops, 0, buf);
		return std::string(buf);
	}
	
	/** @return gmp result */
	std::string get_gmp() {
		// Allocate buffer
		char buf[mpz_sizeinbase(gops[0], 16)+2];
		
		// Print and return string
		mpz_get_str(buf, 16, gops[0]);
		return std::string(buf);
	}
	
	kilo::mpv kops;
	kilo::u32 l;
	mpz_t gops[3];
}; // struct arith_test

int main(int argc, char** argv) {
	
	arith_test test;
	std::string line;
	
	int tests_total = 0, tests_passed = 0;
	
	while ( std::getline(std::cin, line) ) {
		if ( line.size() == 0 ) continue;
		if ( '#' == line[0] ) continue;
		
		std::stringstream ss(line);
		std::string op, val1, val2, kres, gres;
		
		ss >> op;
		ss >> val1;
		ss >> val2;
		
		switch ( op[0] ) {
		case '+': test.add(val1, val2); break;
		case '-': test.sub(val1, val2); break;
		case '*': test.mul(val1, val2); break;
		case '/': test.div(val1, val2); break;
		default: std::cout << "SYNTAX ERROR: `" << line << "'" << std::endl; continue;
		}
		
		kres = test.get_kilomp();
		gres = test.get_gmp();
		
		++tests_total;
		if ( kres == gres ) ++tests_passed;
		else std::cout << "EXPECTED " << gres << " GOT " << kres
		               << " FOR " << val1 << " " << op << " " << val2 << std::endl;
	}
	
	std::cout << tests_passed << "/" << tests_total << " tests passed" << std::endl;
	
	return 0;
}

