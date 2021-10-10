#pragma once

/** 
 * Header file for the simplex algorithm for the KSimplex project.
 */

// Copyright 2013 Aaron Moss
//
// This file is part of KSimplex.
//
// KSimplex is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published 
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// KSimplex is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KSimplex.  If not, see <https://www.gnu.org/licenses/>.


#include <ostream>

#include "ksimplex.hpp"

namespace ksimplex {

/**
 * Solves a linear program using the simplex method. Pivots the given tableau to an optimal or 
 * unbounded basis before returning.
 *  
 * @param Tab			The type of the tableau to use for solving
 * @param tableau		The tableau to pivot to an optimal or unbounded basis
 * @return the last pivot: one of tableau_optimal or tableau_unbounded
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

/**
 * Solves a linear program using the simplex method. Pivots the given tableau to an optimal or 
 * unbounded basis before returning.
 *  
 * @param Tab			The type of the tableau to use for solving
 * @param tableau		The tableau to pivot to an optimal or unbounded basis
 * @param count			Will store number of simplex pivots
 * @return the last pivot: one of tableau_optimal or tableau_unbounded
 */
template<typename Tab>
pivot simplexSolve(Tab& tableau, u32* count) {
	// Get first pivot
	pivot p = tableau.ratioTest();
	
	u32 c = 0;
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

/**
 * Solves a linear program using the simplex method. Pivots the given tableau to an optimal or 
 * unbounded basis before returning.
 *  
 * @param Tab			The type of the tableau to use for solving
 * @param tableau		The tableau to pivot to an optimal or unbounded basis
 * @param count			Will store number of simplex pivots
 * @param out			Print stream to print pivots as we see them
 * @return the last pivot: one of tableau_optimal or tableau_unbounded
 */
template<typename Tab>
pivot simplexSolve(Tab& tableau, u32* count, std::ostream& out) {
	// Get first pivot
	pivot p = tableau.ratioTest();
	
	u32 c = 0;
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

} /* namespace ksimplex */

