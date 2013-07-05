/**
 * Fuzz tester for kilomp library.
 * Generates test cases for test.cpp.
 *
 * @author Aaron Moss
 */

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include <gmp.h>

/** Maximum number of bits in the random numbers */
static const int MAX_BITS = 128;

/** Generate a random number */
void rand_gen(mpz_t& op, gmp_randstate_t& rs) {
	int bits = rand() % MAX_BITS;         // Pick a random number of bits up to `MAX_BITS`
	mpz_urandomb(op, rs, bits);           // Generate a random number with at most `bits` bits
	if ( rand() & 0x1 ) mpz_neg(op, op);  // Negate the number half the time
}

std::string to_string(mpz_t& op) {
	char buf[mpz_sizeinbase(op, 16)+2];   // Allocate a sufficiently large buffer
	mpz_get_str(buf, 16, op);             // Print into it
	return std::string(buf);              // And return
}

/**
 * Usage: fuzz <n>
 * <n> is the number of tests to generate for each of '+', '-', '*', and '/'
 */
int main(int argc, char** argv) {
	using namespace std;
	
	if ( argc <= 1 ) {
		cerr << "Usage: " << argv[0] << " <n>" << endl;
		return 1;
	}
	
	int n = atoi(argv[1]);
	
	// Initialize random seeds for generation
	srand(time(NULL));
	gmp_randstate_t rs;
	gmp_randinit_default(rs);
	gmp_randseed_ui(rs, time(NULL));
	
	// Initialize mp_ints for use
	mpz_t op1, op2;
	mpz_init(op1); mpz_init(op2);
	
	// Generate random numbers
	for (int i = 0; i < n; ++i) {
		// Generate random values
		rand_gen(op1, rs); rand_gen(op2, rs);
		
		// Get strings
		string s1 = to_string(op1), s2 = to_string(op2);
		
		// Print '+', '-', and '*' cases
		cout << "+ " << s1 << " " << s2 << "\n"
		     << "- " << s1 << " " << s2 << "\n"
		     << "* " << s1 << " " << s2 << endl;
		
		// Generate '/' case for exact division and print
		mpz_mul(op1, op1, op2);
		s1 = to_string(op1);
		
		cout << "/ " << s1 << " " << s2 << endl;
	}
	
	// Free memory
	mpz_clear(op1); mpz_clear(op2);
	gmp_randclear(rs);
	
	return 0;
}

