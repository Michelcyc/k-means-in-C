#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "help.h"
#include "kmeans.h"

int main(int argc, char **argv) {
	if (argc < 6) {
		puts("Not enough parameters...");
		exit(1);
	}
	const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
	if ((n < 1) || (m < 1) || (k < 1) || (k > n)) {
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

	clock_t cl = clock();
	kmeans(x, y, n, m, k);
	cl = clock() - cl;
	if (argc > 6) {
		int *ideal = (int*)malloc(n * sizeof(int));
		if (ideal == NULL) {
			fprintf_result(argv[5], y, n);
		} else {
			fscanf_splitting(argv[6], ideal, n);
			const double a = get_accuracy(ideal, y, n);
			printf("Accuracy of k-means clustering = %.5lf;\n", a);
			fprintf_full_result(argv[5], y, n, a);
			free(ideal);
		}
	} else {
		fprintf_result(argv[5], y, n);
	}
	printf("Time for k-means clustering = %.6lf s.;\nThe work of the program is completed...\n", (double)cl / CLOCKS_PER_SEC);

	free(x);
	free(y);
	return 0;
}
