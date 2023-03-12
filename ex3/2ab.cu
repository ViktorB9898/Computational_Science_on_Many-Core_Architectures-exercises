#include <stdio.h>
#include <iostream>
#include "timer.hpp"
#include "cuda_errchk.hpp" // for error checking of CUDA calls
#include <vector>
#include <algorithm>
    
__global__
void transpose(double *A, int N)
{
    int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int row_idx = t_idx / N;
    int col_idx = t_idx % N;
    if (row_idx < N && col_idx < N){
        int tmp;
        tmp = A[col_idx * N + row_idx];
        A[col_idx * N + row_idx] = A[row_idx * N + col_idx];
        A[row_idx * N + col_idx] = tmp;
    }
    
}

void print_A(double *A, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i * N + j] << ", ";
    }
    std::cout << std::endl;
    }
}
    
int main(void)
{
    std::vector<double> N_vec = {16,32,64,128,256,512,1024,2048,4096,8192,16384};
    std::vector<double> bandwidth;

    double *A, *cuda_A;
    Timer timer;
    int reps = 6;

    for (const auto &N : N_vec){
        std::vector<double> time_vec;
        
        // Allocate host memory and initialize
        A = (double*)malloc(N*N*sizeof(double));
        
        for (int i = 0; i < N*N; i++) {
        A[i] = i;
        }
        
        //print_A(A, N);
        
        // Allocate device memory and copy host data over
        CUDA_ERRCHK(cudaMalloc(&cuda_A, N*N*sizeof(double))); 
        
        // copy data over
        CUDA_ERRCHK(cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice));
        
        // repetitions
        for (int j = 0; j < reps; j++){
            // wait for previous operations to finish, then start timings
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();
            
            // Perform the transpose operation
            transpose<<<(N*N+255)/256, 256>>>(cuda_A, N);
            
            // wait for kernel to finish, then print elapsed time
            CUDA_ERRCHK(cudaDeviceSynchronize());
            time_vec.push_back(timer.get());
        }
        std::sort(time_vec.begin(), time_vec.end());
        bandwidth.push_back(4*N*N*sizeof(double)/time_vec[reps/2]*1e-9);
        
        // copy data back (implicit synchronization point)
        CUDA_ERRCHK(cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost));
        
        //print_A(A, N);

        cudaFree(cuda_A);
        free(A);
    }
    
    printf("Gb/s:\n");
    for (const auto& value : bandwidth){
        std::cout << value << "," << std::endl;
    }

    CUDA_ERRCHK(cudaDeviceReset());  // for CUDA leak checker to work
    
    return EXIT_SUCCESS;
}
    

