#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MSG_SIZE 1024
#define NUM_LEADERS 2 // Número de líderes en cada nodo

void generate_random_data(char *buf, int size) {
    for (int i = 0; i < size; i++) {
        buf[i] = 'A' + (rand() % 26); 
    }
}

void binary_tree_comm(char *data, int size, MPI_Comm comm) {
    int rank, comm_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    int left_child = 2 * rank + 1; 
    int right_child = 2 * rank + 2; /

    
    if (rank != 0) {
        int parent = (rank - 1) / 2; 
        MPI_Recv(data, size, MPI_CHAR, parent, 0, comm, MPI_STATUS_IGNORE);
    }

    
    if (left_child < comm_size) {
        MPI_Send(data, size, MPI_CHAR, left_child, 0, comm);
    }
    if (right_child < comm_size) {
        MPI_Send(data, size, MPI_CHAR, right_child, 0, comm);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs, sharedmemRank, sharedmemSize, bridgeCommSize, leader;
    MPI_Comm comm, sharedmemComm, bridgeComm;
    MPI_Win sharedmemWin;
    char *s_buf, *r_buf;
    double start_time, end_time;

    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedmemComm);
    MPI_Comm_rank(sharedmemComm, &sharedmemRank);
    MPI_Comm_size(sharedmemComm, &sharedmemSize);

    // División de líderes 
    leader = (sharedmemRank < NUM_LEADERS) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(comm, leader, rank, &bridgeComm);

    if (bridgeComm != MPI_COMM_NULL) {
        MPI_Comm_size(bridgeComm, &bridgeCommSize);
    }

    
    MPI_Win_allocate_shared((sharedmemRank == 0) ? MSG_SIZE * nprocs : 0,
                            sizeof(char), MPI_INFO_NULL, sharedmemComm, &s_buf, &sharedmemWin);
    if (sharedmemRank != 0) {
        int disp_unit;
        MPI_Aint size;
        MPI_Win_shared_query(sharedmemWin, 0, &size, &disp_unit, &s_buf);
    }

  
    r_buf = (char *)malloc(MSG_SIZE * nprocs);

    
    if (rank == 0) {
        generate_random_data(s_buf, MSG_SIZE);
    }

    
    MPI_Barrier(comm);
    start_time = MPI_Wtime();

    // Comunicación entre nodos usando árbol binario
    if (bridgeComm != MPI_COMM_NULL) {
        binary_tree_comm(s_buf, MSG_SIZE, bridgeComm);
        MPI_Barrier(sharedmemComm); 
    }

    
    if (sharedmemRank < NUM_LEADERS) {
        MPI_Bcast(s_buf, MSG_SIZE, MPI_CHAR, 0, sharedmemComm);
    } else {
        MPI_Barrier(sharedmemComm);
    }

    MPI_Barrier(comm);
    end_time = MPI_Wtime();

    
    if (rank == 0) {
        printf("Tiempo total: %f segundos\n", end_time - start_time);
    }

    free(r_buf);
    MPI_Win_free(&sharedmemWin);
    MPI_Finalize();
    return 0;
}
