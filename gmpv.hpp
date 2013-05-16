#pragma once

#include "ksimplex.hpp"

/// Helper functions for vectors of GMP integers

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
