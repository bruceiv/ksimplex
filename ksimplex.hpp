#pragma once

#include "kilomp/fixed_width.hpp"

/** 
 * Shared definitions for the KSimplex project
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
