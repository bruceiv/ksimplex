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
