/** LRS-based preprocessor to format input nicely for sump (because it's easier 
 *  than rebuilding it myself...)
 * 
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <iostream>

#include "sump.hpp"
#include "sump_io.hpp"
#include "lrspp.hpp"

#include "lrs/matrix.hpp"

int main(int argc, char** argv) {
	using namespace sump;
	
	ind n, d;
	
	// Read in size and dimension of problem
	std::cin >> n;
	std::cin >> d;
	
	// Read in matrix, grab objective row first
	lrs::vector_mpq& obj = *parseVector(std::cin, d);
	lrs::matrix_mpq& mat = *parseMatrix(std::cin, n, d);
	
	// Create LRS instance for pre-processing, and move it to an initial basis
	lrs::lrs& l = *makeLRS(mat);
	l.getFirstBasis();
	
	// Set objective into LRS instance (if you do this before initial basis, 
	//LRS solves the LP itself, which is no fun...)
	l.setLPObj(obj);
	
	// Write size and dimension of problem, as well as determinant
	std::cout << n << " " << d << " " 
		<< convertMpz(l.getDeterminant()) << std::endl;
	
	// Write initial basis
	lrs::ind* bas = l.getLPBasis();
	index_list b = convertIndexList(bas, n);
	writeIndexList(std::cout, b, n); std::cout << std::endl;
	
	// Write objective/constraint matrix
	chimp::chimpz* m = convertMat(l, n, d);
	writeMatrix(std::cout, m, n, d);
	
	// Cleanup
	freeMat(m);
	freeIndexList(b);
	delete[] bas;
	delete &l;
	delete &mat;
	delete &obj;
	
	return 0;
}
