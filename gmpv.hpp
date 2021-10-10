#pragma once

#include "ksimplex.hpp"

/**
 * Helper functions for vectors of GMP integers
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

/** @return a freshly initialized vector of mpz_t of length n */
static mpz_t* init_gmpv(u32 n) {
	mpz_t* v = new mpz_t[n];
	for (u32 i = 0; i < n; ++i) { mpz_init(v[i]); }
	return v;
}

/** Copies src (length n) into dst */
static void copy_gmpv(mpz_t* dst, mpz_t* src, u32 n) {
	for (u32 i = 0; i < n; ++i) { mpz_set(dst[i], src[i]); }
}

/** Frees a vector of mpz_t */
static void clear_gmpv(mpz_t* v, u32 n) {
	for (u32 i = 0; i < n; ++i) { mpz_clear(v[i]); }
	delete[] v;
}

} /* namespace ksimplex */
