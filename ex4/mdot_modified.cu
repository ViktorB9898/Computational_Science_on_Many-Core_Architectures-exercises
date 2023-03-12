#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include "timer.hpp"
#include "cuda_errchk.hpp"

int main(void)
{
  std::vector<size_t> N_vec = {1000,5000,10000,50000,100000,500000,1000000};
  std::vector<double> times;
  Timer timer;
  int reps = 6;

  for (const auto &N : N_vec){
    std::vector<double> time_vec;
    const size_t K = 8;

    //
    // Initialize CUBLAS:
    //
    std::cout << "Init CUBLAS..." << std::endl;
    cublasHandle_t h;
    cublasCreate(&h);


    //
    // allocate host memory:
    //
    std::cout << "Allocating host arrays..." << std::endl;
    double  *x = (double*)malloc(sizeof(double) * N);
    double **y = (double**)malloc(sizeof(double*) * K);
    for (size_t i=0; i<K; ++i) {
      y[i] = (double*)malloc(sizeof(double) * N);
    }
    double *results  = (double*)malloc(sizeof(double) * K);
    double *results2 = (double*)malloc(sizeof(double) * K);


    //
    // allocate device memory
    //
    std::cout << "Allocating CUDA arrays..." << std::endl;
    double *cuda_x; cudaMalloc( (void **)(&cuda_x), sizeof(double)*N);
    double **cuda_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
    for (size_t i=0; i<K; ++i) {
      cudaMalloc( (void **)(&cuda_y[i]), sizeof(double)*N);
    }

    //
    // fill host arrays with values
    //
    for (size_t j=0; j<N; ++j) {
      x[j] = 1 + j%K;
    }
    for (size_t i=0; i<K; ++i) {
      for (size_t j=0; j<N; ++j) {
        y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
      }
    }

    //
    // Reference calculation on CPU:
    //
    for (size_t i=0; i<K; ++i) {
      results[i] = 0;
      results2[i] = 0;
      for (size_t j=0; j<N; ++j) {
        results[i] += x[j] * y[i][j];
      }
    }    
   
    //
    // Copy data to GPU
    //
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
    for (size_t i=0; i<K; ++i) {
      cudaMemcpy(cuda_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
    }


    //
    // Let CUBLAS do the work:
    //

    // repetitions
    for (int j = 0; j < reps; j++){
      // wait for previous operations to finish, then start timings
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();

      for (int i = K / 8; i > 0; i--)
      {
        for (size_t i=0; i<K; ++i) {
          cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
        }
      }

      CUDA_ERRCHK(cudaDeviceSynchronize());
      time_vec.push_back(timer.get());
    }
    std::sort(time_vec.begin(), time_vec.end());
    times.push_back(time_vec[reps/2]);

    std::cout << "Running dot products with CUBLAS..." << std::endl;


    //
    // Compare results
    //
    std::cout << "Copying results back to host..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      std::cout << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
    }

    
    //
    // Clean up:
    //
    std::cout << "Cleaning up..." << std::endl;
    free(x);
    cudaFree(cuda_x);

    for (size_t i=0; i<K; ++i) {
      free(y[i]);
      cudaFree(cuda_y[i]);
    }
    free(y);
    free(cuda_y);

    free(results);
    free(results2);
 
    cublasDestroy(h);
  }
  std::cout << "\ntime [s]:\n" << std::endl;
  for (const auto& value : times){
      std::cout << value << "," << std::endl;
  }
  return 0;
}
