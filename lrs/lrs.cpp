/** Implements lrs::lrs class from lrs.hpp C++ wrapper for LRS.
 *
 *  @author Aaron Moss
 */

#include <cstdio>
#include <new>
#include <sstream>
#include <utility>

#include <gmpxx.h>

#include "clrs.hpp"
#include "cobasis.hpp"
#include "lrs.hpp"
#include "matrix.hpp"

namespace lrs {
	
	lrs::lrs(matrix_mpq const& m, index_set const& lin, lrs_opts o) : o(o) {
		/* Initialize LRS */
		lrs_init_quiet(stdin, stderr);
		
		/* Init LRS global data */
		Q = lrs_alloc_dat((char*)"LRS globals");
		if (Q == 0) throw std::bad_alloc();
		
		/* Init LRS LP dictionary */
		initDat(Q, m.size(), m.dim());
		
		P = lrs_alloc_dic(Q);
		if (P == 0) throw std::bad_alloc();
		
		initDic(Q, P, m, lin);
	}
	
	lrs::lrs(matrix_mpq const& m, index_set const& lin, val_t& det, const ind* bas, 
	         lrs_opts o) : o(o) {
		/* Initialize LRS */
		lrs_init_quiet(stdin, stderr);
		
		/* Init LRS global data */
		Q = lrs_alloc_dat((char*)"LRS globals");
		if (Q == 0) throw std::bad_alloc();
		
		/* Init LRS LP dictionary */
		initDat(Q, m.size(), m.dim());
		
		P = lrs_alloc_dic(Q);
		if (P == 0) throw std::bad_alloc();
		
		initDic(Q, P, m, lin);
		
		// Copy determinant value in
		copy(P->det, det);
		
		// Generate new row, column, basis & cobasis structures
		for (ind i = 0; i <= P->m; ++i) P->Row[i] = i;
		P->Col[P->d] = 0;
		for (ind i = 0; i < P->d; ++i) P->Col[i] = i+1;
		
		ind i_b = 0, i_c = 0;
		for (ind i = 0; i <= P->m + P->d + 1; ++i) {
			if ( i_b <= P->m && i == bas[i_b] ) {
				P->B[i_b] = i;
				++i_b;
			} else {
				P->C[i_c] = i;
				++i_c;
			}
		}
		
		// Initialize the inequality array
		ind i = 1;
		ind lastdv = Q->lastdv;
		for (i = lastdv + 1; i <= P->m; ++i) Q->inequality[P->B[i]-lastdv] = i;
		for (i = 0; i < lastdv; ++i) Q->inequality[P->C[i]-lastdv] = i;
	}
	
	lrs::~lrs() {
		lrs_free_dic(P, Q);
		lrs_free_dat(Q);
		lrs_close_quiet();
	}
	
	index_set lrs::allRatio(ind leave) {
		
		/* assign local variable to structures */
		matrix_t& A = P->A;
		ind* B = P->B;
		ind* Row = P->Row;
		ind* Col = P->Col;
		ind* minratio = Q->minratio;
		ind m = P->m;
		ind d = P->d;
		ind lastdv = Q->lastdv;
		
		ind cob, col;
		
		
		if ( (cob = findCob(leave)) < 0 ) 
			throw lrs_error(
				"Failed to find cobasis index for leaving index" + leave);
		
		col = Col[cob];
		
		ind degencount = 0;
		
		for (ind j = lastdv + 1; j <= m; j++) {
			/* search slack rows with negative coefficient in dictionary for 
			 * leaving columns; minratio is a temp for indices of min ratio 
			 * cols */
			if ( negative(A[Row[j]][col]) ) minratio[degencount++] = j;
		}
		
		mpz_class Nmin, Dmin;
		
		ind start = 0;			/* starting location in minratio array */
		ind bindex = d + 1;		/* index of next basic variable to consider */
		ind cindex = 0;			/* index of next cobasic variable to consider */
		ind basicindex = d;		/* index of basis inverse for current ratio 
								 * test, except d=rhs test */
		bool firstTime = true;	/* For ratio test, true on first pass, else 
								 * false */
		
		ind nstart = 0, ndegencount = 0;
		
		if ( B[bindex] == basicindex ) { /* VOODOO todo check safe removal */
			/* identity col in basis inverse */
			
			if ( minratio[start] == bindex ) {
				/* remove this index, all others stay */
				start++;
				degencount--;
			}
			bindex++;
			
		} else {
			/* perform ratio test on rhs or column of basis inverse */
			
			/* get next ratio column and increment cindex */
			for (ind j = start; j < start + degencount; j++) {
				ind i = Row[minratio[j]];	/* i is the row location of the 
											 * next basic variable */
				int comp = 1;	/* 1: lhs>rhs; 0: lhs=rhs; -1: lhs<rhs */
				
				if (firstTime) {
					firstTime = false; /* force new min ratio on first time */
				} else { 
					/* compare test = A[i][0]/A[i][col] vs. min = Nmin/Dmin */
					
					/* keep in mind here that A[i][col] < 0 (by construction), 
					 * therefore Dmin < 0 as well. */
					
					/* min < 0 or test > 0 */
					if ( sgn(Nmin) > 0 || negative( A[i][0] ) ) {
						comp = 
							/* min > 0 or test < 0 i.e. sgn(min) == sgn(test) */
							( sgn(Nmin) < 0 || positive( A[i][0] ) ) ?
							/* compare(A[i][0]/A[i][col],Nmin/Dmin) */
							comprod( Nmin.get_mpz_t(), A[i][col], 
									 A[i][0], Dmin.get_mpz_t() )
							/* min < 0 and test > 0 */
							: -1;
					/* min == 0 and test == 0 */
					} else if ( sgn(Nmin) == 0 && zero( A[i][0] ) ) {
						comp = 0;
					}
					
					/* all signs reversed for rhs */
					comp = -comp;
				}
				
				if ( comp == 1 ) { /* new minimum ratio */
					nstart = j;
					Nmin = mpz_class( A[i][0] );
					Dmin = mpz_class( A[i][col] );
					ndegencount = 1;
				} else if ( comp == 0 ) { /* repeated minimum */
					minratio[nstart + ndegencount++] = minratio[j];
				}
			}
		}
		
		degencount = ndegencount;
		start = nstart;
		
		/* prepare return set */
		index_set rval(m+d+1);
		for (ind i = start; i < start + degencount; i++) {
			rval.set(B[minratio[i]]);
		}
		
		return rval;
	}
	
	index_set lrs::arrangementRatio(ind leave) {
		
		/* assign local variable to structures */
		matrix_t& A = P->A;
		ind* B = P->B;
		ind* Row = P->Row;
		ind* Col = P->Col;
		ind* minratio = Q->minratio;
		ind m = P->m;
		ind d = P->d;
		ind lastdv = Q->lastdv;
		
		ind cob, col;
		
		if ( (cob = findCob(leave)) < 0 ) 
			throw lrs_error(
				"Failed to find cobasis index for leaving index" + leave);
		
		col = Col[cob];
		
		/* Indices of next possible negative/positive/zero entering variables */
		ind nEnter = 0, pEnter = m+1, zEnter = 0;
		/* minimum negative/positive numerators/denominators */
		mpz_t nNmin, nDmin, pNmin, pDmin;
		mpz_init(nNmin); mpz_init(nDmin); mpz_init(pNmin); mpz_init(pDmin);
		/* First negative/positive/zero value found */
		bool nFirst = true, pFirst = true;
		
		for (ind j = lastdv + 1; j <= m; j++) {
			/* search slack rows with non-zero coefficient in dictionary for 
			 * leaving columns */
			ind i = Row[j];
			int comp = 1; /* 1: new min, 0: repeated min, -1: not min */
			int signFound = 0; /* 1: positive, -1: negative, 0: zero */
			bool negDen = false; /* true for A[i][col] < 0 */
			
			/* These, in general, compare tmp = A[i][0]/A[i][col] vs. 
			 * nmin = nNmin/nDmin or pmin = pNmin/pDmin. I use the pseudocode 
			 * operator a <=> b to denote a comparison operation that returns 
			 * 1 for a > b, 0 for a == b, or -1 for a < b. Also, throughout, I 
			 * normalize the minimum values such that the denominator is 
			 * positive, so that the cross-multiplying trick for comparing 
			 * rationals works (a/b <=> c/d == ad <=> bc if b, d > 0) (Note 
			 * also that a/b <=> c/d == -(ad <=> bc) if b < 0, d > 0). This 
			 * algorithm will also take all zero ratios (degenerate pivots).
			 */
			
			if ( positive( A[i][col] ) ) {
				
				if ( positive( A[i][0] ) ) {
					/* positive ratio */
					signFound = 1;
					if ( pFirst ) { 
						/* new minimum */
						pFirst = false; 
					} else {
						/* comp = pmin <=> tmp */
						comp = comprod(pNmin, A[i][col], A[i][0], pDmin);
					}
				} else if ( negative( A[i][0] ) ) {
					/* negative ratio */
					signFound = -1;
					if ( nFirst ) {
						/* new minimum */
					} else {
						/* comp = tmp <=> nmin */
						comp = comprod(A[i][0], nDmin, nNmin, A[i][col]);
					}
				}  else /* if ( zero( A[i][0] ) ) */ {
					/* zero ratio */
					comp = 0;
				}
				
			} else if ( negative( A[i][col] ) ) {
				negDen = true;
				
				if ( negative( A[i][0] ) ) {
					/* positive ratio */
					signFound = 1;
					if ( pFirst ) { 
						/* new minimum */
						pFirst = false; 
					} else {
						/* comp = pmin <=> tmp */
						comp = comprod(A[i][0], pDmin, pNmin, A[i][col]);
					}
				} else if ( positive( A[i][0] ) ) {
					/* negative ratio */
					signFound = -1;
					if ( nFirst ) {
						/* new minimum */
					} else {
						/* comp = tmp <=> nmin */
						comp = comprod(nNmin, A[i][col], A[i][0], nDmin);
					}
				} else /* if ( zero( A[i][0] ) ) */ {
					/* zero ratio */
					comp = 0;
				}
				
			} /* otherwise this is an invalid pivot, with a zero in the 
			   * denominator column */
			
			ind tmp;
			if ( comp == 1 ) { /* new minimum */
				switch ( signFound ) {
					case 1:
						/* set new minratio, ensuring positive denominator */
						if ( negDen ) {
							mpz_neg(pNmin, A[i][0]);
							mpz_neg(pDmin, A[i][col]);
						} else {
							mpz_set(pNmin, A[i][0]);
							mpz_set(pDmin, A[i][0]);
						}
						/* store value at the end of the array */
						minratio[m] = j; pEnter = m-1;
						break;
					case -1:
						/* set new minratio, ensuring positive denominator */
						if ( negDen ) {
							mpz_neg(nNmin, A[i][0]);
							mpz_neg(nDmin, A[i][col]);
						} else {
							mpz_set(nNmin, A[i][0]);
							mpz_set(nDmin, A[i][0]);
						}
						/* store value at array start (after zeros) */
						minratio[zEnter] = j; nEnter = zEnter + 1;
						break;
				}
			} else if ( comp == 0 ) { /* repeated minimum */
				switch ( signFound ) {
					case 1:
						/* store value at array end */
						minratio[pEnter--] = j;
						break;
					case 0:
						/* push negative start forward */
						minratio[nEnter++] = minratio[zEnter];
						/* store value at array start (before negatives) */
						minratio[zEnter++] = j;
					case -1:
						/* store value at array start (after zeros) */
						minratio[nEnter++] = j;
						break;
				}
			}
		}
		
		mpz_clear(nNmin); mpz_clear(nDmin); mpz_clear(pNmin); mpz_clear(pDmin);
		
		/* prepare return set */
		index_set rval(m+d+1);
		for (ind i = 0; i < nEnter; i++) {
			/* read in zeros and min negatives */
			rval.set( B[minratio[i]] );
		}
		for (ind i = pEnter+1; i <= m; i++) {
			/* read in positive values */
			rval.set( B[minratio[i]] );
		}
		
		return rval;
	}
	
	std::pair<ind,ind> lrs::blandRatio() {
		matrix_t& A = P->A;
		ind* B = P->B;
		ind* C = P->C;
		ind* Row = P->Row;
		ind* Col = P->Col;
		ind lastdv = Q->lastdv;
		ind m = P->m;
		ind d = P->d;
		
		ind col, row;
		ind enter = 0, leave = m+d+1;
		
		/* find mimimum positive objective value */
		for (ind k = 0; k < d; ++k) {
			if ( positive(A[0][Col[k]]) && C[k] < leave ) {
				col = Col[k];
				leave = C[k];
			}
		}
		
		/* tableau optimal */
		if ( leave == m+d+1 ) return std::make_pair<ind,ind>(0, 0);
		
		/* find minratio leaving column */
		for (ind k = lastdv + 1; k <= m; ++k) {
			if ( negative(A[Row[k]][col]) ) {
				if ( enter == 0 ) {
					row = Row[k];
					enter = B[k];
				} else {
					ind i = B[k];
					
					/* Compute Bland's ratio */
					ind comp = comprod(A[row][0], A[Row[k]][col], A[Row[k]][0], A[row][col]);
					
					/* Test ratio */
					if ( comp == -1 || ( comp == 0 && i < enter ) ) {
						row = Row[k];
						enter = i;
					}
				}
			}
		}
		
		/* tableau unbounded */
		if ( enter == 0 ) return std::make_pair<ind,ind>(0xFFFFFFFF, 0xFFFFFFFF);
		
		return std::make_pair(enter,leave);
	}
	
	std::pair<ind,ind> lrs::danRatio() {
		ind* B = P->B;
		ind* C = P->C;
		ind lastdv = Q->lastdv;
		ind d = P->d;
		
		ind enter, leave;
		if ( ::dan_selectpivot(P, Q, &enter, &leave) ) {
			/* pivot found */
			return std::make_pair(B[enter], C[leave]);
		} else {
			/* tableau optimal or unbounded */
			if ( leave == d ) {
				/* tableau optimal */
				return std::make_pair<ind,ind>(0, 0);
			} else {
				/* tableau unbounded */
				return std::make_pair<ind,ind>(0xFFFFFFFF, 0xFFFFFFFF);
			}
		}
	}
	
	ind lrs::findBas(ind enter) {
		ind j;
		ind lastdv = Q->lastdv;
		ind *B = P->B;
		ind m = P->m;
		
		for (j = lastdv+1; j <= m && B[j] != enter; j++);
		
		return (B[j] != enter) ? -1 : j;
	}
	
	ind lrs::findCob(ind leave) {
		ind j;
		ind lastdv = Q->lastdv;
		ind *C = P->C;
		ind d = P->d;
		
		for (j = 0; j < d && C[j] != leave; j++);
		
		return (C[j] != leave) ? -1 : j;
	}
	
	cobasis* lrs::getCobasis(ind col) {
		/* Ported from David Bremner's equivalent code in lrsserv.c */
		
		ind i;
		ind rflag = -1;  /* for finding inequality number of ray column */
		
		/* set local variables for structures */
		matrix_t& A = P->A;
		ind* B = P->B;
		ind* C = P->C;
		ind* Col = P->Col;
		ind* Row = P->Row;
		ind* tempArray = Q->temparray;
		ind d = P->d;
		ind lastdv = Q->lastdv;
		ind m = P->m;
		
		ind nIncidence;  /* count number of tight inequalities */
		
		index_set cobInd(m+d+1);
		index_set extraInc(m+d+1);
		
		for (i = 0; i < d; i++) {
			if (Col[i] == col) rflag = tempArray[i]; /* look for ray index */
			
			tempArray[i] = C[i];
			cobInd.set(tempArray[i]);
		}
		
		nIncidence = (col == 0) ? d : d-1;
		
		for (i = lastdv + 1; i <= m; i++) {
			if ( zero( A[Row[i]][0] ) ) {
				if ( (col == 0L) || zero( A[Row[i]][col] ) ) {
					extraInc.set(B[i]);
					nIncidence++;
				}
			}
		}
		
		mpz_class det(P->det);
		
		cobasis* cob = new cobasis(det, rflag, cobInd, nIncidence, extraInc);
		return cob;
	}
	
	bool lrs::getFirstBasis(bool solve) {
		/* Lin is an out parameter of this method, so it isn't initialized */
		return solve ? 
			lrs_getfirstbasis(&P, Q, &Lin, true) :
			lrs_getfirstbasis_nosolve(&P, Q, &Lin, true);
	}
	
	ind lrs::getRealSize() {
		return P->m;
	}
	
	ind lrs::getRealDim() {
		return P->d;
	}
	
	vector_mpz* lrs::getSolution(ind col) {
		
		if (col < 0 || col > P->d) {
			std::ostringstream err;
			err << "getSolution: illegal column " << col;
			throw lrs_error( err.str() );
		}
		
		vector_mpz* output_p = new vector_mpz(Q->n);
		
		if (! lrs_getsolution(P, Q, output_p->v, col) ) {
			delete output_p;
			output_p = 0;
		}
		
		return output_p;
	}
	
	vector_mpz* lrs::getVertex() {
		ind i  = 1, iRedund = 0;
		
		ind nRedundCol = Q->nredundcol;
		ind* redundCol = Q->redundcol;
		ind n = Q->n;
		
		vector_mpz& output = *new vector_mpz(n);
		
		/* copy column 0 to output */
		copy( output[0], P->det );
		
		for (ind j = 1; j < n; j++) {
			if (iRedund < nRedundCol && redundCol[iRedund] == j) {
				/* column was deleted as redundant */
				itomp(0L, output[j]);
				iRedund++;
			} else {
				/* column not deleted as redundant */
				getnextoutput(P, Q, i, 0L, output[j]);
				i++;
			}
		}
		
		reducearray(output.v, n);
		
		return &output;
	}
	
	void lrs::initDat(lrs_dat* Q, ind n, ind d) {
		
		if (o.vRepresentation) {
			/* V-representation */
			Q->hull = true;
			Q->polytope = true;
			/* NOTE Symbal appears to always turn on geometric, LRS ignores it 
			 * for V-representation. I'm following Symbal here. */
			Q->geometric = true;
		} else {
			/* H-representation */
			Q->hull = false;
			
			/* copied from Symbal, equivalent to setting the geometric option 
			 * in LRS */
			Q->geometric = true;
		}
		
		Q->m = n; Q->n = d;
		
	}
	
	void lrs::initDic(lrs_dat* Q, lrs_dic* P, matrix_mpq const& mat, 
					  index_set const& lin) {
		
		ind m = Q->m;
		
		/* read matrix row by row */
		matrix_mpq& m_nc = const_cast<matrix_mpq &>(mat);
		for (ind i = 0; i < m; i++) {
			vector_mpq_base row = m_nc[i];
			lrs_set_row_mp(
				P, Q, i+1, row.num().v, row.den().v, (lin[i+1]) ? eq : ge);
		}
		
	}
	
	ind lrs::lexRatio(ind leave) {
		ind cob;
		
		ind* Col = P->Col;
		ind* B = P->B;
		ind lastdv = Q->lastdv;
		
		if ( ( cob = findCob(leave) ) < 0 ) {
			return -1;
		}
		
		ind col = Col[cob];
		ind enter = ratio(P, Q, col);
		
		return ( enter > 0 ) ? B[enter] : -1;
	}
	
	void lrs::pivot(ind enter, ind leave) {
		ind bas = findBas(enter);
		if (bas < 0) throw lrs_error("Failed to find basis for pivot.");
		
		ind cob = findCob(leave);
		if (cob < 0) throw lrs_error("Failed to find cobasis for pivot.");
		
		::pivot(P, Q, bas, cob);
		::update(P, Q, &bas, &cob);
	}
	
	void lrs::printDict() {
		/* Copied the code up to lrs.cpp to use iostreams, rather than trying 
		 * to mix iostreams and stdio on the same output stream */
// 		printA(P, Q);
		ind i, j;
		
		matrix_t& A = P->A;
		ind* B = P->B;
		ind* C = P->C;
		ind* Row = P->Row;
		ind* Col = P->Col;
		ind m = P->m, d = P->d, lastdv = Q->lastdv;
		
		std::ostream& out = o.output();
		
		out << "\n Basis    "; for (i = 0; i <= m; i++) out << B[i] << " ";
		out << " Row "; for (i = 0; i <= m; i++) out << Row[i] << " ";
		
		out << "\n Co-Basis "; for (i = 0; i <= d; i++) out << C[i] << " ";
		out << " Column "; for (i = 0; i <= d; i++) out << Col[i] << " ";
		out << " det=" << toString(P->det);
//		out << "\n Ineq     "; for (i = 0; i <=m; ++i) out << Q->inequality[i] << " ";
		out << "\n";
		
		for (i = 0; i <= m; i++) {
			out << "A[" << B[i] << "]";
			for (j = 0; j <= d; j++) {
				out << "[" << C[j] << "]= " << toString(A[Row[i]][Col[j]]) 
					<< " ";
			}
			
			if (i == 0 && Q->nonnegative) { /* skip basic rows - don't exist! */
				i = d;
			}
			
			out << std::endl;
		}
	}
	
	void lrs::setCobasis(index_set& cob) {
		ind nlinearity = Q->nlinearity;
		ind* linearity = Q->linearity;
		ind* facet = Q->facet;
		ind m = Q->m;
		ind d = P->d;
		ind j = nlinearity;
		index_set_iter k = begin(cob);
		
		for ( ; j < d && k != end(cob); ++j, ++k) {
			facet[j] = *k;
			
			/* check errors TODO replace with asserts? Boost asserts? */
			if ( facet[j] < 1 || facet[j] > m ) {
				std::ostringstream err;
				err << "Start/restart cobasic indices must be in range [1," 
					<< m << "]";
				throw lrs_error(err.str());
			}
			for (ind i = 0; i < nlinearity; i++) {
				if ( facet[j] == linearity[i] ) {
					throw lrs_error(
						"Start/restart cobasic indices should not include "
						"linearities");
				}
			}
			for (ind i = 0; i < j; i++) {
				if ( facet[i] == facet[j] ) {
					
					throw lrs_error(
						"Start/restart cobasic indices must be distinct");
				}
			}
		}
			
		Q->restart=true;
		if ( !restartpivots(P, Q) )
			throw lrs_error("Could not restart pivots from given cobasis");
		
	}
	
	val_t& lrs::elem(ind i, ind j) {
		return P->A[P->Row[i]][P->Col[j]];
	}
		
	val_t& lrs::getDeterminant() {
		return P->det;
	}
	
	ind* lrs::getLPCobasis() {
		ind* C = P->C;
		ind d = P->d;
		
		ind* lpc = new ind[d];
		for (ind i = 0; i < d; i++) lpc[i] = C[i];
		return lpc;
	}
	
	ind* lrs::getLPBasis() {
		ind* B = P->B;
		ind m = P->m;
		
		ind* lpb = new ind[m];
		for (ind i = 1; i <= m; i++) lpb[i-1] = B[i];
		return lpb;
	}
	
	void lrs::setLPObj(vector_mpq const& v) {
		/* copy in objective row */
		vector_mpq& vec = const_cast<vector_mpq&>(v);
		lrs_set_obj_mp(P, Q, vec.num().v, vec.den().v, 1);
		
		/* multiply through by determinant */
		val_t& det = P->det;
		for (ind j = Q->hull; j <= P->d; ++j) {
			mulint(P->A[0][j], det, P->A[0][j]);
		}
	}
	
	std::string lrs::toString(val_t& x) {
		std::stringstream s;
		if ( ! negative(x) ) s << " ";
		s << mpz_class(x);
		return s.str();
	}
	
} /* namespace lrs */
