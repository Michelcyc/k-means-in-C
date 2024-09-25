#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "help.h"
#include "kmeans.h"

struct timeval start, end;

int main(int argc, char **argv) {
    //Checa os parametros
	if (argc < 6) {
		puts("Not enough parameters...");
		exit(1);
	}
	//Atribui valores para as variaveis n=n_amostras, m=n_features, k=n_clusters e checa erros
	const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
	if ((n < 1) || (m < 1) || (k < 1) || (k > n)) {
		puts("Values of input parameters are incorrect...");
		exit(1);
	}
	//Array x sera uma array 1D com todo o dataframe
	double *x = (double*)malloc(n * m * sizeof(double));
	if (x == NULL) {
		puts("Memory allocation error...");
		exit(1);
	}
	//Array y sera uma array 1D com o cluster associado ao elemento
	int *y = (int*)malloc(n * sizeof(int));
	if (y == NULL) {
		puts("Memory allocation error...");
		free(x);
		exit(1);
	}
	//Aqui o arquivo de teste eh convertido pra dentro da memoria no formato de array 1D, x
	fscanf_data(argv[1], x, n * m);

	//clock_t cl = clock();
	//kmeans(x, y, n, m, k);
	//cl = clock() - cl;

	gettimeofday(&start, NULL);
    kmeans(x, y, n, m, k);
    gettimeofday(&end, NULL);

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
	double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Time for k-means clustering = %.6lf s\n", elapsed_time);

	free(x);
	free(y);
	return 0;
}
