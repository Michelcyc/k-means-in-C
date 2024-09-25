#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


double get_distance(const double *x1, const double *x2, int m);
void autoscaling(double* const x, const int n, const int m);
char constr(const int *y, const int val, int s);
void det_cores(const double* const x, double* const c, const int n, const int m, const int k);
int get_cluster(const double* const x, const double* const c, const int m, int k);
void det_start_splitting(const double *x, const double *c, int* const y, int n, const int m, const int k);
char check_splitting(const double *x, double *c, int* const res, const int n, const int m, const int k);
void kmeans(const double* const X, int* const y, const int n, const int m, const int k);

#endif
