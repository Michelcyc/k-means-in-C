#include <stdio.h>
#include <stdlib.h>

#include <omp.h> // Include OpenMP for omp_get_wtime()
#include "help.h"
#include "kmeans.h"

int main(int argc, char **argv) {
	if (argc < 6) {
		puts("Not enough parameters...");
		exit(1);
	}
	const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]), th = atoi(argv[6]);
	if ((n < 1) || (m < 1) || (k < 1) || (k > n) || (th < 0)) {
		puts("Values of input parameters are incorrect...");
		exit(1);
	}
	double *x = (double*)malloc(n * m * sizeof(double));
	if (x == NULL) {
		puts("Memory allocation error...");
		exit(1);
	}
	int *y = (int*)malloc(n * sizeof(int));
	if (y == NULL) {
		puts("Memory allocation error...");
		free(x);
		exit(1);
	}
	fscanf_data(argv[1], x, n * m);

	double start_time = omp_get_wtime();
	kmeans(x, y, n, m, k, th);
	double end_time = omp_get_wtime();

	fprintf_result(argv[5], y, n);
	printf("Time for k-means clustering = %.6lf s.;\nThe work of the program is completed...\n", end_time - start_time);

	free(x);
	free(y);
	return 0;
}
