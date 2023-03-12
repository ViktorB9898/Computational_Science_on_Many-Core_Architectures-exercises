#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[256];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}


int main()
{
    std::vector<int> N_vec = {100,500,1000,5000,10000,50000,100000,500000};
    std::vector<double> bandwidth;
    std::vector<double> times;
    Timer timer;
    int reps = 6;

    for (const auto &N : N_vec){
        std::vector<double> time_vec;

        // Allocate and initialize arrays on CPU

        double *x = (double *)malloc(sizeof(double) * N);
        double *y = (double *)malloc(sizeof(double) * N);
        double result = 0;

        std::fill(x, x + N, -2);
        // add a zero to x
        std::fill(y, y + N, 2);

        // Allocate and initialize arrays on GPU

        double *cuda_x;
        double *cuda_y;
        double *cuda_result;

        CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
        CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
        CUDA_ERRCHK(cudaMalloc(&cuda_result, sizeof(double)));

        CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
          CUDA_ERRCHK(cudaMemcpy(cuda_result, &result, sizeof(double), cudaMemcpyHostToDevice));

        // repetitions
        for (int j = 0; j < reps; j++){
            // wait for previous operations to finish, then start timings
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();

            cuda_dot_product<<<256, 256>>>(N, cuda_x, cuda_y, cuda_result);

            CUDA_ERRCHK(cudaDeviceSynchronize());
            time_vec.push_back(timer.get());
        }
        std::sort(time_vec.begin(), time_vec.end());
        times.push_back(time_vec[reps/2]);
        bandwidth.push_back(2*N*sizeof(double)/time_vec[reps/2]*1e-9);

        CUDA_ERRCHK(cudaMemcpy(&result, cuda_result, sizeof(double), cudaMemcpyDeviceToHost));

        // Clean up

        CUDA_ERRCHK(cudaFree(cuda_x));
        CUDA_ERRCHK(cudaFree(cuda_y));
        CUDA_ERRCHK(cudaFree(cuda_result));
        
        free(x);
        free(y);
    }

    std::cout << "Gb/s:\n" << std::endl;
    for (const auto& value : bandwidth){
        std::cout << value << "," << std::endl;
    }

    std::cout << "\ns:\n" << std::endl;
    for (const auto& value : times){
        std::cout << value << "," << std::endl;
    }



    CUDA_ERRCHK(cudaDeviceReset()); // for CUDA leak checker to work

    return EXIT_SUCCESS;
}
