#ifndef _SUMP_SIMPLEX_HPP_
#define _SUMP_SUMP_HPP_
/** Header file for the simplex algorithm for the "Simplex Using 
 *  Multi-Precision" (sump) project.
 *  
 *  @author Aaron Moss (moss.aaron@unb.ca)
 */

#include <ostream>

#include "sump.hpp"

namespace sump {
	/** Solves a linear program using the simplex method. Pivots the given tableau 
	 *  to an optimal or unbounded basis before returning.
	 *  
	 *  @param Tab		The type of the tableau to use for solving
	 *  @param tableau	The tableau to pivot to an optimal or unbounded basis
	 *  @return the last pivot: one of tableau_optimal or tableau_unbounded
	 */
	template<typename Tab>
	pivot simplexSolve(Tab& tableau) {
		// Get first pivot
		pivot p = tableau.ratioTest();
		
		// Pivot as long as more pivots exist
		while ( p != tableau_optimal && p != tableau_unbounded ) {
			tableau.doPivot(p.enter, p.leave);
			p = tableau.ratioTest();
		}
		
		// Return last pivot
		return p;
	}
	
	/** Solves a linear program using the simplex method. Pivots the given 
	 *  tableau to an optimal or unbounded basis before returning.
	 *  
	 *  @param Tab		The type of the tableau to use for solving
	 *  @param tableau	The tableau to pivot to an optimal or unbounded basis
	 *  @param count	Will store number of simplex pivots
	 *  @return the last pivot: one of tableau_optimal or tableau_unbounded
	 */
	template<typename Tab>
	pivot simplexSolve(Tab& tableau, ind* count) {
		// Get first pivot
		pivot p = tableau.ratioTest();
		
		ind c = 0;
		// Pivot as long as more pivots exist
		while ( p != tableau_optimal && p != tableau_unbounded ) {
			tableau.doPivot(p.enter, p.leave);
			++c;
			p = tableau.ratioTest();
		}
		*count = c;
		
		// Return last pivot
		return p;
	}
	
	/** Solves a linear program using the simplex method. Pivots the given 
	 *  tableau to an optimal or unbounded basis before returning.
	 *  
	 *  @param Tab		The type of the tableau to use for solving
	 *  @param tableau	The tableau to pivot to an optimal or unbounded basis
	 *  @param count	Will store number of simplex pivots
	 *  @param out		Print stream to print pivots as we see them
	 *  @return the last pivot: one of tableau_optimal or tableau_unbounded
	 */
	template<typename Tab>
	pivot simplexSolve(Tab& tableau, ind* count, std::ostream& out) {
		// Get first pivot
		pivot p = tableau.ratioTest();
		
		ind c = 0;
		// Pivot as long as more pivots exist
		while ( p != tableau_optimal && p != tableau_unbounded ) {
			out << "(" << p.leave << "," << p.enter << ")" << std::endl;
			tableau.doPivot(p.enter, p.leave);
			++c;
			p = tableau.ratioTest();
		}
		*count = c;
		
		// Return last pivot
		return p;
	}
}


#endif /* _SUMP_SIMPLEX_HPP_ */
