#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>


void generate_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (double) rand() / RAND_MAX * 5.0;  
    }
}


void print_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}


void bpmf(int m, int n, int k, double* R, double* W, double* H, int iterations, int rank, int size) {
    int block_size = m / size;  

    
    double* local_R = (double*) malloc(block_size * n * sizeof(double));
    double* local_W = (double*) malloc(block_size * k * sizeof(double));
    double* local_H = (double*) malloc(k * n * sizeof(double));

    
    MPI_Scatter(R, block_size * n, MPI_DOUBLE, local_R, block_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    MPI_Bcast(W, m * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(H, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    for (int iter = 0; iter < iterations; ++iter) {
        
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < k; ++j) {
                double sum = 0.0;
                for (int l = 0; l < n; ++l) {
                    sum += (local_R[i * n + l] - local_W[i * k + j] * H[j * n + l]) * H[j * n + l];
                }
                local_W[i * k + j] += 0.01 * sum; 
            }
        }

        
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < n; ++l) {
                double sum = 0.0;
                for (int i = 0; i < block_size; ++i) {
                    sum += (local_R[i * n + l] - local_W[i * k + j] * H[j * n + l]) * local_W[i * k + j];
                }
                H[j * n + l] += 0.01 * sum; 
            }
        }

        
        MPI_Allreduce(MPI_IN_PLACE, W, m * k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Bcast(H, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    
    free(local_R);
    free(local_W);
    free(local_H);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = 4, n = 4, k = 2; 
    int iterations = 10;

    
    if (m % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "El número de procesos debe ser un divisor de m (dimensión de las matrices).\n");
        }
        MPI_Finalize();
        return -1;
    }

    
    double* R = (double*) malloc(m * n * sizeof(double));
    double* W = (double*) malloc(m * k * sizeof(double));
    double* H = (double*) malloc(k * n * sizeof(double));

    
    if (rank == 0) {
        generate_matrix(R, m, n);
        for (int i = 0; i < m * k; ++i) W[i] = 1.0;
        for (int i = 0; i < k * n; ++i) H[i] = 1.0;

        printf("Matriz R:\n");
        print_matrix(R, m, n);
    }

    
    bpmf(m, n, k, R, W, H, iterations, rank, size);

    
    if (rank == 0) {
        printf("\nMatriz W:\n");
        print_matrix(W, m, k);

        printf("\nMatriz H:\n");
        print_matrix(H, k, n);
    }

    
    free(R);
    free(W);
    free(H);

    MPI_Finalize();
    return 0;
}
