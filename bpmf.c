#include <iostream>
#include <vector>
#include <random>


void generate_matrix(std::vector<double>& mat, int rows, int cols) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 5.0);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = distribution(generator);
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


void bpmf(int m, int n, int k, std::vector<double>& R, std::vector<double>& W, std::vector<double>& H, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        

        for (int i = 0; i < m; ++i) {
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
                for (int i = 0; i < m; ++i) {
                    sum += (R[i * n + l] - W[i * k + j] * H[j * n + l]) * W[i * k + j];
                }
                H[j * n + l] += 0.01 * sum; 
            }
        }
    }
}

int main() {
    int m = 4, n = 4, k = 2; 
    int iterations = 10;

    std::vector<double> R(m * n), W(m * k, 1.0), H(k * n, 1.0);
    generate_matrix(R, m, n);

    std::cout << "Matriz R:\n";
    print_matrix(R, m, n);

    bpmf(m, n, k, R, W, H, iterations);

    std::cout << "\nMatriz W:\n";
    print_matrix(W, m, k);

    std::cout << "\nMatriz H:\n";
    print_matrix(H, k, n);

    return 0;
}
