#pragma once

#include "kilomp/fixed_width.hpp"

/** 
 * Shared definitions for the KSimplex project
 * 
 * @author Aaron Moss
 */

namespace ksimplex {
	
//import fixed width types into this namespace
using kilo::u32;
using kilo::s32;
using kilo::u64;
using kilo::s64;

/** Represents a simplex pivot */
struct pivot {
	u32 enter;  ///< entering variable
	u32 leave;  ///< leaving variable
	
	pivot(u32 e, u32 l) : enter(e), leave(l) {}
	pivot(const pivot& that) : enter(that.enter), leave(that.leave) {}
	
	pivot& operator = (const pivot& that) {
		enter = that.enter;
		leave = that.leave;
		return *this;
	}
	
	bool operator == (const pivot& that) { return enter == that.enter && leave == that.leave; }
	bool operator != (const pivot& that) { return enter != that.enter || leave != that.leave; }
}; /* struct pivot */

/// Special pivot to indicate that the tableau is optimal
static const pivot tableau_optimal = pivot(0, 0);

/// Special pivot to indicate that the tableau is unbounded
static const pivot tableau_unbounded = pivot(0xFFFFFFFF, 0xFFFFFFFF);

} /* namespace ksimplex */
