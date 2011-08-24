
#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#define MAX_DIFF 0.000001

int main ( int argc, char ** argv ) 
{

	FILE * f1 = fopen ( argv[1], "r" );
	FILE * f2 = fopen ( argv[2], "r" );
	int i = 0;

    double maxdiff = 0;

	if ( f1 == NULL || f2 == NULL ) {printf("wrong files\n");exit ( -1 );}
	
	
	while ( !feof ( f1 ) ) {
		double firstValue, secondValue, diff;
		
		fscanf ( f1, "%lf", &firstValue );
		fscanf ( f2, "%lf", &secondValue );
		
		diff = firstValue - secondValue;

		//printf ( "%lf  <--> %lf (diff = %lf, fabs=%lf)\n", firstValue, secondValue, diff, fabs(diff) );

		//		printf ( "%lf  ", diff );
		//		if ( abs(diff) >= MAX_DIFF ) {
		if ( fabs(diff) > MAX_DIFF ) {
		  printf ( "%d: (%lf <--> %lf) : diff = %lf ", i, firstValue, secondValue, diff );
			printf ( " --> TOO MUCH\n" );
		}

        if ( fabs( diff ) > fabs( maxdiff ) ) {
          maxdiff = diff;
        }
		i++;
		//else 
		//printf ( "\n" );
	}

    printf( "maximum difference: %lf\n", maxdiff );
	
	exit ( 0 );
}
