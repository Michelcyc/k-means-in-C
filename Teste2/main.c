/*
 * Copyright 2020 SPeCS.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
 * an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License. under the License.
 *
 * Nuno Paulino
 * nuno.m.paulino@inesctec.pt
 * INESC TEC, Porto, PORTUGAL
 * Created: 19-09-2018
 * Modified: 14-07-2020
 */

#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iomanip>
#include <math.h>
#include <limits.h>
#include <float.h>

#define DATATYPE double
#define str(s) #s
#define FORMATER "%.2d"
#define DATATYPEMAX DBL_MAX

#define KMEANS_ERROR_TOLERANCE 1  //error tolerance for kmeans
int t = KMEANS_ERROR_TOLERANCE;

static int *labels;
static DATATYPE *points;
static DATATYPE *centroids;
static DATATYPE *cs;
static DATATYPE *c1;
static DATATYPE *gt;
static int *counts;
static DATATYPE *min_distance;
static int itcount[1] = {0};

void dataprep(char *datafile, int n, int nfeatures, int nclusters) {

	///////////////////////////////////////////////////////////////////////////
	// initialize the array of the data points to be pre-classified: database of classified points
	// generate random centroids

	FILE *data = fopen(datafile, "r");
	if(!data) {
		printf("Can't open data file!\n");
		exit(1);
	}

	for(int i=0; i < n; i++) {
		for(int j = 0; j < nfeatures; j++) {
			int x;
			fscanf(data, " %d", &x);
			points[(i * nfeatures) + j] = (DATATYPE) x;
		}
	}
	fflush(data);
	fclose(data);

	// gt
	char datafilegt[80];
	strcpy(datafilegt, datafile);
	char *ptr = strstr(datafilegt, ".txt");
	strcpy(ptr, "_gt.txt");

	FILE *datagt = fopen(datafilegt, "r");
	if(!datagt) {
		printf("Can't open ground truth file!\n");
		exit(1);
	}

	for(int i=0; i < nclusters; i++) {
		for(int j = 0; j < nfeatures; j++) {
			int x;
			fscanf(datagt, " %d", &x);
			gt[(i * nfeatures) + j] = (DATATYPE) x;
		}
	}
	fflush(datagt);
	fclose(datagt);

	// pick k points as initial centroids
	srand(123);

	DATATYPE *dists = (DATATYPE*) calloc(n, sizeof(DATATYPE));
	double *cumsum = (double*) calloc(n, sizeof(double));
	int i, j;
	for (i = 0; i < nclusters; i++) {
		unsigned int idx = ((((float)rand())/((float)RAND_MAX)) * n);
		int maxdist = 0;
		double fidx = 0.0f;

		if(i == 0)
			goto skip;

		// check distance to all centroids already chosen
		for (int k1 = 0; k1 < n; k1++) {
			dists[k1] = DATATYPEMAX;
			cumsum[k1] = 0;
			for (int k2 = 0; k2 < i; k2++) {
				DATATYPE tmp = 0;
				for (int k3 = 0; k3 < nfeatures; k3++)
					tmp += (fabs(centroids[k2 * nfeatures + k3] -
						points[k1 * nfeatures + k3]));

				if(tmp < dists[k1])
					dists[k1] = tmp;
			}
		}

		// maxdist?
		maxdist = 0;
		for (int k1 = 0; k1 < n; k1++)
			maxdist += dists[k1];

		// compute commulative sum
		cumsum[0] = ((float)dists[0] / (float)maxdist);
		for (int k1 = 1; k1 < n; k1++) {
			cumsum[k1] = cumsum[k1-1] + ((float)dists[k1] / (float)maxdist);
			//printf("%f\n", cumsum[k1]);
		}

		// generate random number between 0 and 1
		fidx = ((((float)rand())/((float)RAND_MAX)) * cumsum[n-1]);

		// select new centroid
		for (int k1 = 0; k1 < n-1; k1++) {
			if(fidx >= cumsum[k1] && fidx <= cumsum[k1+1]) {
				idx = k1;
				break;
			}
		}
skip:
		for (j = 0; j < nfeatures; j++)
			centroids[i * nfeatures + j] = points[idx * nfeatures + j];
	}
	free(dists);
	free(cumsum);

	// save initial centroids
	memcpy(cs, centroids, nclusters * nfeatures * sizeof(double));
	return;
}

void kmeans(
	DATATYPE *data,
	int n, int m, int k, int t,
	DATATYPE *centroids,
	int *labels,
	DATATYPE *c1,
	int *counts,
	int *itcount)
{
    int h, i, j = 0;
    double old_error, error = DBL_MAX;
    itcount[0] = 0;

    // main loop
    do {
        // save error from last step
        old_error = error, error = 0;

        // clear old counts and temp centroids
        for (i = 0; i < k; i++) {
        	counts[i] = 0;
            for (j = 0; j < m; j++)
            	c1[i * m  + j] = 0;
        }

        for (h = 0; h < n; h++) {

            // identify the closest cluster
        	double min_distance = DBL_MAX;
            for (i = 0; i < k; i++) {
                double distance = 0;
                for (j = 0; j < m; j++) {
                	double diff = (data[h * m + j] - centroids[i * m + j]);
                	distance += diff*diff;
                }
                if ((double)(distance/2) < (double)(min_distance/2)) {
                    labels[h] = i;
                    min_distance = distance;
                }
            }

            // update size and temp centroid of the destination cluster
            counts[labels[h]]++;
            for (j = 0; j < m; j++)
            	c1[labels[h] * m + j] += data[h * m + j];

            // update standard error
            error += min_distance;
		}
		itcount[0]++;
		// update all centroids
		for (i = 0; i < k; i++)
			if(counts[i] > 0)
				for (j = 0; j < m; j++)
					centroids[i * m + j] = c1[i * m + j] / counts[i];

    } while(fabs((error - old_error)) > t);
}

void printout(int fullrun, char *datafile, int n, int nfeatures, int nclusters) {

	int i = 0, j = 0;

	char newname[240];
	sprintf(newname, "../results/clusters_cpu_%s_%dpoints_%dfeatures_%dclusters.m",
		"double", n, nfeatures, nclusters);

	FILE *fd = fopen(newname, "w");

	fprintf(fd, "cs = [");
	for(i = 0; i < nclusters; i++) {
		for(j = 0; j < nfeatures-1; j++) {
			fprintf(fd, "%f, ", cs[(nfeatures * i) + j]);
		}
		fprintf(fd, "%f", cs[(nfeatures * i) + j]);
		fprintf(fd, ";\n");
	}
	fprintf(fd, "];\n");

	if(fullrun == 0) {
		fflush(fd);
		fclose(fd);
		return;
	}

	fprintf(fd, "a = [");
	for(i = 0; i < n; i++) {
		for(j = 0; j < nfeatures; j++) {
			fprintf(fd, ""FORMATER", ", (unsigned int) points[(nfeatures * i) + j]);
		}
		fprintf(fd, "%d;\n", labels[i]);
	}
	fprintf(fd, "];\n");

	fprintf(fd, "c = [");
	for(i = 0; i < nclusters; i++) {
		for(j = 0; j < nfeatures-1; j++) {
			fprintf(fd, "%f, ", centroids[(nfeatures * i) + j]);
		}
		fprintf(fd, "%f", centroids[(nfeatures * i) + j]);
		fprintf(fd, ";\n");
	}
	fprintf(fd, "];\n");

	fprintf(fd, "gt = [");
	for(i = 0; i < nclusters; i++) {
		for(j = 0; j < nfeatures-1; j++) {
			fprintf(fd, ""FORMATER", ", (unsigned int) gt[(nfeatures * i) + j]);
		}
		fprintf(fd, ""FORMATER"", (unsigned int) gt[(nfeatures * i) + j]);
		fprintf(fd, ";\n");
	}
	fprintf(fd, "];\n");

	fprintf(fd, "clustersizes = [");
	for(i = 0; i < nclusters-1; i++)
		fprintf(fd, ""FORMATER", ", counts[i]);
	fprintf(fd, ""FORMATER"];\n", counts[nclusters-1]);

	fflush(fd);
	fclose(fd);
	return;
}

int main(int argc, char* argv[])
{
	if(argc != 6 || (strcmp(argv[1], "help") == 0)) {
		printf("Usage: %s inital/cluster <number of points> "
		"<number of clusters> <number of attributes> <filename>", argv[0]);
		return -1 ;
	}

	int fullrun = 0;
	if(strcmp(argv[1], "inital") == 0) {
		fullrun = 0;

	} else if(strcmp(argv[1], "cluster") == 0) {
		fullrun = 1;

	} else {
		return 0;
	}

	const int n = atoi(argv[2]);
	const int nclusters = atoi(argv[3]);
	const int nfeatures = atoi(argv[4]);

	labels = (int *) calloc(n, sizeof(int));
	points = (DATATYPE *) calloc(n * nfeatures, sizeof(DATATYPE));
	centroids = (DATATYPE *) calloc(nclusters * nfeatures, sizeof(DATATYPE));
	cs = (DATATYPE *) calloc(nclusters * nfeatures, sizeof(DATATYPE));
	gt = (DATATYPE *) calloc(nclusters * nfeatures, sizeof(DATATYPE));
	c1 = (DATATYPE *) calloc(nfeatures * nclusters, sizeof(DATATYPE));
	counts = (int *) calloc(nclusters, sizeof(int));
	min_distance = (DATATYPE *) calloc(n, sizeof(DATATYPE));

	dataprep(argv[5], n, nfeatures, nclusters);

	if(fullrun == 1) {

		clock_t start = clock();

		kmeans(points, n, nfeatures, nclusters, t,
			centroids, labels, c1, counts, &itcount[0]);

		clock_t end = clock();
		float nanoseconds = ((float)(end - start) / CLOCKS_PER_SEC) * 1E9;
		printf("runtime = %f\n", nanoseconds);
	}

	printout(fullrun, argv[4], n, nfeatures, nclusters);

	free(points);
	free(centroids);
	free(cs);
	free(labels);
	free(c1);
	free(gt);
	free(counts);
	free(min_distance);

	return 0;
}

