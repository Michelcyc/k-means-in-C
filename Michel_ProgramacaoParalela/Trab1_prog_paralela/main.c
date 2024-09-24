#include <stdio.h>
#include <stdlib.h>


int main()
{
    int n_clusters;
    printf("Insira o numero de clusters\n");
    if (scanf("%d", &n_clusters) == 1) {
        printf("You entered the integer: %d\n", n_clusters);

        double x1[] = {1.0, 2.0};
        double x2[] = {4.0, 6.0};
        int dimensions = 2;

        // Call the get_distance function
        double distance_squared = get_distance(x1, x2, dimensions);

        // Print the result
        printf("The squared Euclidean distance between x1 and x2 is: %.2f\n", distance_squared);



    } else {
        printf("Input invalido.\n");
    }
    return 0;
}
