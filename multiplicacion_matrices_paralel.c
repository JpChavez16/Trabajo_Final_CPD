#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void generate_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (double) rand() / RAND_MAX;
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


void summa(int n, double* A, double* B, double* C, int rank, int size) {
    int block_size = n / size;  

    
    double* local_A = (double*) malloc(block_size * n * sizeof(double));
    double* local_C = (double*) malloc(block_size * n * sizeof(double));

    
    MPI_Scatter(A, block_size * n, MPI_DOUBLE, local_A, block_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

 
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    MPI_Gather(local_C, block_size * n, MPI_DOUBLE, C, block_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    free(local_A);
    free(local_C);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 4; 

    
    if (n % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "El número de procesos debe ser un divisor de n (dimensión de las matrices).\n");
        }
        MPI_Finalize();
        return -1;
    }

   
    double* A = (double*) malloc(n * n * sizeof(double));
    double* B = (double*) malloc(n * n * sizeof(double));
    double* C = (double*) malloc(n * n * sizeof(double));

    
    if (rank == 0) {
        generate_matrix(A, n, n);
        generate_matrix(B, n, n);
        
        printf("Matriz A:\n");
        print_matrix(A, n, n);

        printf("\nMatriz B:\n");
        print_matrix(B, n, n);
    }

    
    summa(n, A, B, C, rank, size);

    
    if (rank == 0) {
        printf("\nResultado (C = A x B):\n");
        print_matrix(C, n, n);
    }

    
    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
