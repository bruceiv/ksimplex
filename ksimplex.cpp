/**
 * Main driver for the KSimplex project.
 * 
 * @author Aaron Moss
 */

#include <iostream>
#include <string>

#include "kmp_tableau.hpp"
#include "ksimplex.hpp"
#include "simplex.hpp"

#include "kilomp/kilomp.cuh"

using namespace ksimplex;

/// Number of limbs to initially allocate in the matrix
static const u32 INIT_ALLOC = 4;

/**
 * Reads a word into a vector index
 * 
 * @param m			The vector to read in to
 * @param i			The index in the vector to write the new value to
 * @param m_l		The number of elements in m
 * @param a_l		The number of currently allocated limbs (may be updated)
 * @param u_l		The number of currently used limbs (may be updated)
 * @param in		The stream to read from
 */
void parse(kilo::mpv m, u32 i, u32 m_l, u32& a_l, u32& u_l, std::istream& in) {
	// Read the next word
	std::string s;
	in >> s;
	
	// Ensure the vector has enough space to accomodate it
	u32 len = s.size();
	u32 l = (len+7)/8;
	if ( l > a_l ) {
		m = kilo::expand(m, m_l, a_l, 2*l);
		a_l = 2*l;
	}
	
	// Parse the value
	u32 u = kilo::parse(m, i, s.c_str(), len);
	if ( u > u_l ) u_l = u;
}

/**
 * Reads a matrix from input
 * 
 * @param m			The matrix to read in to
 * @param n			The number of rows in the matrix (not counting objective)
 * @param d			The number of columns in the matrix (not counting constant)
 * @param a_l		The number of currently allocated limbs (may be updated)
 * @param u_l		The number of currently used limbs (may be updated)
 * @param in		The stream to read from
 */
void parseMatrix(kilo::mpv m, u32 n, u32 d, u32& a_l, u32& u_l, 
                 std::istream& in) {
	u32 m_l = 1 + (n+1)+(d+1);
	for (u32 i = 1; i < m_l; ++i) { parse(m, i, m_l, a_l, u_l, std::cin); }
}

/**
 * Prints a vector index.
 * 
 * @param m			The vector to print from
 * @param i			The index in the vector to print from
 * @param out		The stream to write to
 */
void print(const kilo::mpv m, u32 i, std::ostream& out) {
	//allocate large enough buffer
	u32 len = kilo::size(m, i);
	char buf[8*len+2];
	
	//print into buffer, then stdout
	kilo::print(m, i, buf);
	out << buf;
}

/**
 * Prints a matrix.
 * 
 * @param m			The vector to print from
 * @param n			The number of rows in the matrix (not counting objective)
 * @param d			The number of columns in the matrix (not counting constant)
 * @param out		The stream to write to
 */
void printMatrix(const kilo::mpv m, u32 n, u32 d, std::ostream& out) {
	for (u32 i = 0; i <= n; ++i) {
		for (u32 j = 0; j <= d; ++j) {
			out << " ";
			print(m, 1 + i*(d+1) + j, out);
		}
		out << std::endl;
	}
}

/**
 * Simple test driver.
 * 
 * Input (tableau file):
 * <n> <d> <det>
 * <basis_element>{n}
 * (<tableau_element>{d+1}){n+1}
 */
int main(int argc, char **argv) {
	// Read in size and dimension of the the problem
	u32 n, d, m_l;
	std::cin >> n;
	std::cin >> d;
	m_l = 1 + (n+1)*(d+1);
	
	// Build appropriately large mp-vector to store values
	u32 a_l = INIT_ALLOC, u_l = 0;
	kilo::mpv mat = kilo::init_mpv(m_l, a_l);
	
	// Read in initial determinant
	parse(mat, 0, m_l, a_l, u_l, std::cin);
	
	// Read in basis
	u32* bas = new u32[n+1];
	for (u32 i = 0; i <= n; ++i) { std::cin >> bas[i]; }
	
	// Generate cobasis
	u32* cob = new u32[d+1];
	u32 c_j = 1, b_i = 1, j = 1;
	while ( b_i <= n ) {
		if ( j == bas[b_i] ) { ++j; ++b_i; continue; }
		while ( j < bas[b_i] && c_j <= d ) { cob[c_j] = j; ++c_j; ++j; }
		++b_i;
	}
	while ( j <= n+d && c_j <= d ) { cob[c_j] = j; ++c_j; ++j; }
	
	// Read in matrix
	parseMatrix(mat, n, d, a_l, u_l, std::cin);
	
	// Construct tableau
	kmp_tableau tab(n, d, a_l, u_l, cob, bas, mat);
	
	// Print initial tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Run simplex algorithm
	u32 pivot_count = 0;
	pivot p = simplexSolve(tab, &pivot_count, std::cout);
	
	if ( p == tableau_optimal ) {
		std::cout << "tableau: OPTIMAL" << std::endl;
	} else if ( p == tableau_unbounded ) {
		std::cout << "tableau: UNBOUNDED" << std::endl;
	}
	
	// Print final tableau
	printMatrix(tab.mat(), n, d, std::cout);
	
	// Cleanup
	delete[] cob;
	delete[] bas;
	kilo::clear(mat, a_l);
	
	return 0;
}
