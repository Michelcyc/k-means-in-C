#include "kmeans.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double get_distance(const double *x1, const double *x2, int m) {
    double d, r = 0.0;
    while (m--) {
        d = *(x1++) - *(x2++);
        r += d * d;
    }
    return r;
}

void autoscaling(double* const x, const int n, const int m) {
    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
        double Ex = 0.0, Exx = 0.0;

        // Calculate mean and variance in one loop with reduction
        #pragma omp parallel for reduction(+:Ex, Exx)
        for (int i = 0; i < n; i++) {
            double sd = x[i * m + j];
            Ex += sd;
            Exx += sd * sd;
        }

        Ex /= n;
        Exx /= n;
        double stddev = sqrt(Exx - Ex * Ex);

        // Normalize the column
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i * m + j] = (x[i * m + j] - Ex) / stddev;
        }
    }
}

char constr(const int *y, const int val, int s) {
    while (s--) {
        if (*(y++) == val) return 1;
    }
    return 0;
}

void det_cores(const double* const x, double* const c, const int n, const int m, const int k) {
    int *nums = (int*)malloc(k * sizeof(int));
    srand((unsigned int)time(NULL));
    #pragma omp parallel for
    for (int i = 0; i < k; i++) {
        int val = rand() % n;
        while (constr(nums, val, i)) {
            val = rand() % n;
        }
        nums[i] = val;
        memcpy(c + i * m, x + val * m, m * sizeof(double));
    }
    free(nums);
}

int get_cluster(const double* const x, const double* const c, const int m, int k) {
    int res = --k;
    double minD = get_distance(x, c + k * m, m);
    while (k--) {
        double curD = get_distance(x, c + k * m, m);
        if (curD < minD) {
            minD = curD;
            res = k;
        }
    }
    return res;
}

void det_start_splitting(const double *x, const double *c, int* const y, int n, const int m, const int k) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = get_cluster(x + i * m, c, m, k);
    }
}

int check_splitting(const double *x, double *c, int* const res, const int n, const int m, const int k) {
    double *newCores = (double*)malloc(k * m * sizeof(double));
    memset(newCores, 0, k * m * sizeof(double));
    int *nums = (int*)malloc(k * sizeof(int));
    memset(nums, 0, k * sizeof(int));
    int flag = 0;  // Flag to track cluster changes

    double threshold = 1e-6;  // Convergence threshold for centroid movement

    #pragma omp parallel
    {
        double *local_newCores = (double*)calloc(k * m, sizeof(double));
        int *local_nums = (int*)calloc(k, sizeof(int));

        #pragma omp for
        for (int i = 0; i < n; i++) {
            int f = get_cluster(x + i * m, c, m, k);
            if (f != res[i]) {
                #pragma omp atomic write
                flag = 1;  // Use atomic write to avoid race condition
            }
            res[i] = f;
            local_nums[f]++;
            for (int j = 0; j < m; j++) {
                local_newCores[f * m + j] += x[i * m + j];
            }
        }

        // Use atomic operations instead of critical section to avoid bottlenecks
        #pragma omp for
        for (int i = 0; i < k; i++) {
            #pragma omp atomic
            nums[i] += local_nums[i];
            for (int j = 0; j < m; j++) {
                #pragma omp atomic
                newCores[i * m + j] += local_newCores[i * m + j];
            }
        }

        free(local_newCores);
        free(local_nums);
    }

    // Update centroids and check for movement beyond threshold
    int centroids_moved = 0;
    #pragma omp parallel for reduction(+:centroids_moved)
    for (int i = 0; i < k; i++) {
        if (nums[i] > 0) {
            for (int j = 0; j < m; j++) {
                double new_value = newCores[i * m + j] / nums[i];
                if (fabs(new_value - c[i * m + j]) > threshold) {
                    centroids_moved = 1;  // Centroid moved beyond threshold
                }
                c[i * m + j] = new_value;
            }
        }
    }

    free(newCores);
    free(nums);
    return (flag || centroids_moved);  // Continue if clusters changed or centroids moved
}

void kmeans(const double* const X, int* const y, const int n, const int m, const int k, const int th) {
    #if _OPENMP
        omp_set_num_threads(th);
    #endif
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    double *c = (double*)malloc(k * m * sizeof(double));
    det_cores(x, c, n, m, k);
    det_start_splitting(x, c, y, n, m, k);

    int iterations = 0;
    while (check_splitting(x, c, y, n, m, k)) {
        iterations++;
        printf("Iteration %d\n", iterations);
        if (iterations > 1000000000) {  // Avoid infinite loops with a max iteration limit
            printf("Max iterations reached\n");
            break;
        }
    }

    free(x);
    free(c);
}
