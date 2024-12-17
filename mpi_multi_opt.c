#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 512  


void generate_random_matrix(int *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;  
    }
}


void multiply_matrices(int *A, int *B, int *C, int local_rows, int size) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}


void binary_tree_bcast(int *data, int count, int root, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    int mask = 1;
    while (mask < nprocs) {
        if ((rank & mask) == 0) { 
            int target = rank | mask;  
            if (target < nprocs) {
                MPI_Send(data, count, MPI_INT, target, 0, comm);
            }
        } else {  
            int source = rank & (~mask);
            MPI_Recv(data, count, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);
            break;
        }
        mask <<= 1;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs, local_rows;
    MPI_Comm comm, shared_comm;
    int *A, *B, *C, *local_A, *local_C, *shared_B;
    double start_time, end_time;

    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    
    local_rows = MATRIX_SIZE / nprocs;

    
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared_comm);

    int shared_rank;
    MPI_Comm_rank(shared_comm, &shared_rank);

    
    if (rank == 0) {
        A = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
        C = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
        generate_random_matrix(A, MATRIX_SIZE);
    }
    local_A = (int *)malloc(local_rows * MATRIX_SIZE * sizeof(int));
    local_C = (int *)malloc(local_rows * MATRIX_SIZE * sizeof(int));

    
    MPI_Win shared_window;
    if (shared_rank == 0) {
        MPI_Win_allocate_shared(MATRIX_SIZE * MATRIX_SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL, shared_comm, &shared_B, &shared_window);
        generate_random_matrix(shared_B, MATRIX_SIZE);
    } else {
        int disp_unit;
        MPI_Win_allocate_shared(0, sizeof(int), MPI_INFO_NULL, shared_comm, &shared_B, &shared_window);
        MPI_Win_shared_query(shared_window, 0, NULL, &disp_unit, &shared_B);
    }

    // Comunicación en árbol binario
    binary_tree_bcast(A, MATRIX_SIZE * MATRIX_SIZE, 0, comm);

    
    MPI_Scatter(A, local_rows * MATRIX_SIZE, MPI_INT, local_A, local_rows * MATRIX_SIZE, MPI_INT, 0, comm);

    
    MPI_Barrier(comm);
    start_time = MPI_Wtime();

    
    multiply_matrices(local_A, shared_B, local_C, local_rows, MATRIX_SIZE);

    
    MPI_Gather(local_C, local_rows * MATRIX_SIZE, MPI_INT, C, local_rows * MATRIX_SIZE, MPI_INT, 0, comm);

    
    MPI_Barrier(comm);
    end_time = MPI_Wtime();

    
    if (rank == 0) {
        printf("Tiempo total: %f segundos\n", end_time - start_time);
        free(A);
        free(C);
    }

    
    MPI_Win_free(&shared_window);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
