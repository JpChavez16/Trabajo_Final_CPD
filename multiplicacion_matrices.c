#include <iostream>
#include <vector>
#include <cstdlib>

void generate_matrix(std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

void print_matrix(const std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void summa(int n, const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    int block_size = n; 
    for (int k = 0; k < n; k += block_size) {
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int l = k; l < k + block_size && l < n; ++l) {
                    C[i * n + j] += A[i * n + l] * B[l * n + j];
                }
            }
        }
    }
}

int main() {
    int n = 4;

    std::vector<double> A(n * n), B(n * n), C(n * n, 0.0);
    generate_matrix(A, n, n);
    generate_matrix(B, n, n);

    std::cout << "Matriz A:\n";
    print_matrix(A, n, n);

    std::cout << "\nMatriz B:\n";
    print_matrix(B, n, n);

    summa(n, A, B, C);

    std::cout << "\nResultado (C = A x B):\n";
    print_matrix(C, n, n);

    return 0;
}
