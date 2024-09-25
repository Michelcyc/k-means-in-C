#include "kmeans.h"
#include <omp.h>

double get_distance(const double *x1, const double *x2, int m) {
    double d, r = 0.0;
    while (m--) {
        d = *(x1++) - *(x2++);
        r += d * d;
    }
    return r;
}

void autoscaling(double* const x, const int n, const int m) {
    // Arrays to store Ex and Exx for each feature
    double *Ex_array = (double*)calloc(m, sizeof(double));
    double *Exx_array = (double*)calloc(m, sizeof(double));

    // Calculate Ex and Exx in parallel
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double val = x[i * m + j];
            #pragma omp atomic
            Ex_array[j] += val;
            #pragma omp atomic
            Exx_array[j] += val * val;
        }
    }

    // Normalize data
    for (int j = 0; j < m; j++) {
        Ex_array[j] /= n;
        Exx_array[j] /= n;
        double sd = sqrt(Exx_array[j] - Ex_array[j] * Ex_array[j]);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i * m + j] = (x[i * m + j] - Ex_array[j]) / sd;
        }
    }

    free(Ex_array);
    free(Exx_array);
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
    int i;
    for (i = 0; i < k; i++) {
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
        const double curD = get_distance(x, c + k * m, m);
        if (curD < minD) {
            minD = curD;
            res = k;
        }
    }
    return res;
}

void det_start_splitting(const double *x, const double *c, int* const y, int n, const int m, const int k) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
        y[i] = get_cluster(x + i * m, c, m, k);
    }
}

char check_splitting(const double *x, double *c, int* const res, const int n, const int m, const int k) {
    double *newCores = (double*)calloc(k * m, sizeof(double));
    int *nums = (int*)calloc(k, sizeof(int));
    char flag = 0;

    #pragma omp parallel for reduction(+:newCores[:k*m], nums[:k]) reduction(|:flag)
    for (int i = 0; i < n; i++) {
        int f = get_cluster(x + i * m, c, m, k);
        if (f != res[i]) flag = 1;
        res[i] = f;
        nums[f]++;
        int f_m = f * m;
        for (int j = 0; j < m; j++) {
            newCores[f_m + j] += x[i * m + j];
        }
    }

    // Update centroids
    for (int i = 0; i < k; i++) {
        int f = nums[i];
        if (f > 0) {
            for (int j = 0; j < m; j++) {
                c[i * m + j] = newCores[i * m + j] / f;
            }
        }
    }

    free(newCores);
    free(nums);
    return flag;
}


void kmeans(const double* const X, int* const y, const int n, const int m, const int k) {
    double *x = (double*)malloc(n * m * sizeof(double));
    memcpy(x, X, n * m * sizeof(double));
    autoscaling(x, n, m);
    double *c = (double*)malloc(k * m * sizeof(double));
    det_cores(x, c, n, m, k);
    det_start_splitting(x, c, y, n, m, k);
    while (check_splitting(x, c, y, n, m, k));
    free(x);
    free(c);
}
