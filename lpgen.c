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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** Gets a new random variable */
int nextX() {
	return (rand() & 0x1FF) - 0x100;
}

/** Generates a random LP instance. 
 *  
 *  Arguments: \<n\> \<d\>
 *  n: the number of constraints
 *  d: the number of variables
 *  
 *  Output (lrs format):
 *  \<n\> \<d\>
 *  \<x_0,0\> ... \<x_0,d\>
 *  ...
 *  \<x_n,0\> ... \<x_n,d\>
 *  
 *  (each of the x_i,j is a random value in the range (-256,255))
 */
int main(int argc, char** argv) {
	
	int i, j, n, d;
	
	//parse parameters
	if (argc != 3) {
		printf("usage: %s <n> <d>\n", argv[0]);
		return 1;
	}
	
	n = atoi(argv[1]);
	d = atoi(argv[2]);
	
	//print size line
	printf("%d %d\n", n, d);
	
	//init random number generator
	srand(time(NULL));
	
	//print matrix
	for (i = 0; i <= n; ++i) {
		printf("%d", nextX());
		for (j = 1; j <= d; ++j) {
			printf(" %d", nextX());
		}
		printf("\n");
	}
	
	return 0;
}
