#pragma once

/**
 * Simple timing framework for the KSimplex project.
 * 
 * @author Aaron Moss
 */

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

