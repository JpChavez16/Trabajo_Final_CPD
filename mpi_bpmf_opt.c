#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MSG_SIZE 1024
#define NUM_LEADERS 2
#define M 8     // Filas
#define N 8     // Columnas
#define K 4     
#define ITERATIONS 10

void generate_random_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (rand() % 5) + 1.0;
    }
}


void binary_tree_bcast(double *data, int count, int root, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int mask = 1;
    while (mask < nprocs) {
        if ((rank & mask) == 0) {  
            int target = rank | mask;
            if (target < nprocs) {
                MPI_Send(data, count, MPI_DOUBLE, target, 0, comm);
            }
        } else {  
            int source = rank & (~mask);
            MPI_Recv(data, count, MPI_DOUBLE, source, 0, comm, MPI_STATUS_IGNORE);
            break;
        }
        mask <<= 1;
    }
}

void bpmf_update(int start_row, int end_row, int n, int k, double *R, double *W, double *H) {
    for (int iter = 0; iter < ITERATIONS; ++iter) 
    {
        
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < k; ++j) {
                double sum = 0.0;
                for (int l = 0; l < n; ++l) {
                    sum += (R[i * n + l] - W[i * k + j] * H[j * n + l]) * H[j * n + l];
                }
                W[i * k + j] += 0.01 * sum;
            }
        }

        
        for (int j = 0; j < k; ++j) {
            for (int l = 0; l < n; ++l) {
                double sum = 0.0;
                for (int i = start_row; i < end_row; ++i) {
                    sum += (R[i * n + l] - W[i * k + j] * H[j * n + l]) * W[i * k + j];
                }
                double global_sum;
                MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                H[j * n + l] += 0.01 * global_sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs, sharedmemRank, sharedmemSize;
    MPI_Comm comm, sharedmemComm;
    MPI_Win sharedmemWin;
    double *R, *W, *H, *shared_R;
    double start_time, end_time;

    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedmemComm);
    MPI_Comm_rank(sharedmemComm, &sharedmemRank);
    MPI_Comm_size(sharedmemComm, &sharedmemSize);

    // Crear memoria compartida
    MPI_Win_allocate_shared((sharedmemRank == 0) ? M * N * sizeof(double) : 0,
                            sizeof(double), MPI_INFO_NULL, sharedmemComm, &shared_R, &sharedmemWin);
    if (sharedmemRank != 0) {
        int disp_unit;
        MPI_Aint size;
        MPI_Win_shared_query(sharedmemWin, 0, &size, &disp_unit, &shared_R);
    }

    
    W = (double *)malloc(M * K * sizeof(double));
    H = (double *)malloc(K * N * sizeof(double));

    if (rank == 0) {
        srand(time(NULL));
        generate_random_matrix(shared_R, M, N);
        printf("Matriz R generada:\n");
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%.1f ", shared_R[i * N + j]);
            }
            printf("\n");
        }
    }

    // Distribuir en árbol binario
    MPI_Barrier(comm);
    binary_tree_bcast(shared_R, M * N, 0, comm);

    
    start_time = MPI_Wtime();

    
    int rows_per_proc = M / nprocs;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == nprocs - 1) ? M : start_row + rows_per_proc;

    bpmf_update(start_row, end_row, N, K, shared_R, W, H);

    MPI_Barrier(comm);
    end_time = MPI_Wtime();

    
    if (rank == 0) {
        printf("\nMatriz W actualizada:\n");
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                printf("%.2f ", W[i * K + j]);
            }
            printf("\n");
        }

        printf("\nMatriz H actualizada:\n");
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%.2f ", H[i * N + j]);
            }
            printf("\n");
        }

        printf("Tiempo total: %f segundos\n", end_time - start_time);
    }

    free(W);
    free(H);
    MPI_Win_free(&sharedmemWin);
    MPI_Finalize();
    return 0;
}
