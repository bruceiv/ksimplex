#pragma once

/**
 * Simple timing framework for the KSimplex project.
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

#include <ctime>

namespace ksimplex {

typedef std::clock_t timer;

/** @return the current time */
inline timer now() { return std::clock(); }

/** @return the difference between two times */
inline timer ms_between(const timer& begin, const timer& end) {
	return (end - begin) / (CLOCKS_PER_SEC / 1000);
}

} /* namespace ksimplex */

