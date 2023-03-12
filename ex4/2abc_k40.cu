#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void cuda_mdot_product(int N, double *x, double *y, double *res)
{
  // clean res to wipe results from previous repetition
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  {
    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 0;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
  }

  __shared__ double shared_mem1[256]; // remember to only use 256 threads per block then!
  __shared__ double shared_mem2[256];
  __shared__ double shared_mem3[256];
  __shared__ double shared_mem4[256];
  __shared__ double shared_mem5[256];
  __shared__ double shared_mem6[256];
  __shared__ double shared_mem7[256];
  __shared__ double shared_mem8[256];

  double dot1 = 0, dot2 = 0, dot3 = 0, dot4 = 0, dot5 = 0, dot6 = 0, dot7 = 0, dot8 = 0;
  double val_w = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
  {
    // printf("y = %g\n", y[i]);
    val_w = x[i];
    dot1 += val_w * y[i];
    dot2 += val_w * y[i + 1 * N];
    dot3 += val_w * y[i + 2 * N];
    dot4 += val_w * y[i + 3 * N];
    dot5 += val_w * y[i + 4 * N];
    dot6 += val_w * y[i + 5 * N];
    dot7 += val_w * y[i + 6 * N];
    dot8 += val_w * y[i + 7 * N];
  }

  shared_mem1[threadIdx.x] = dot1;
  shared_mem2[threadIdx.x] = dot2;
  shared_mem3[threadIdx.x] = dot3;
  shared_mem4[threadIdx.x] = dot4;
  shared_mem5[threadIdx.x] = dot5;
  shared_mem6[threadIdx.x] = dot6;
  shared_mem7[threadIdx.x] = dot7;
  shared_mem8[threadIdx.x] = dot8;

  for (int k = blockDim.x / 2; k > 0; k /= 2)
  {
    __syncthreads();
    if (threadIdx.x < k)
    {
      __syncthreads();
      shared_mem1[threadIdx.x] += shared_mem1[threadIdx.x + k];
      shared_mem2[threadIdx.x] += shared_mem2[threadIdx.x + k];
      shared_mem3[threadIdx.x] += shared_mem3[threadIdx.x + k];
      shared_mem4[threadIdx.x] += shared_mem4[threadIdx.x + k];
      shared_mem5[threadIdx.x] += shared_mem5[threadIdx.x + k];
      shared_mem6[threadIdx.x] += shared_mem6[threadIdx.x + k];
      shared_mem7[threadIdx.x] += shared_mem7[threadIdx.x + k];
      shared_mem8[threadIdx.x] += shared_mem8[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0)
  {
    __syncthreads();
    res[0] += shared_mem1[0];
    res[1] += shared_mem2[0];
    res[2] += shared_mem3[0];
    res[3] += shared_mem4[0];
    res[4] += shared_mem5[0];
    res[5] += shared_mem6[0];
    res[6] += shared_mem7[0];
    res[7] += shared_mem8[0];
  }
}

int main(void)
{
  std::vector<size_t> N_vec = {1000, 5000, 10000, 50000, 100000, 500000, 1000000};
  std::vector<double> times;
  Timer timer;
  int reps = 6;

  for (const auto &N : N_vec)
  {
    std::vector<double> time_vec;

    // const size_t N = 4000;
    const size_t K = 16;

    //
    // allocate host memory:
    //
    std::cout << "Allocating host arrays..." << std::endl;
    // for x and y
    double *x = (double *)malloc(sizeof(double) * N);
    // use double pointers for vectors (like matrix)
    double **y = (double **)malloc(sizeof(double *) * K / 8);

    // for results on cpu and gpu
    double *results = (double *)malloc(sizeof(double) * K);
    for (size_t i = 0; i < K / 8; ++i)
    {
      y[i] = (double *)malloc(sizeof(double) * N * 8);
    }
    double **results2 = (double **)malloc(sizeof(double) * K / 8);
    for (size_t i = 0; i < K / 8; ++i)
    {
      results2[i] = (double *)malloc(sizeof(double) * 8);
    }
    // for the calculation on the cpu to compare with gpu results
    double **y_cpu = (double **)malloc(sizeof(double *) * K);
    for (size_t i = 0; i < K; ++i)
    {
      y_cpu[i] = (double *)malloc(sizeof(double) * N);
    }

    //
    // allocate device memory
    //
    std::cout << "Allocating CUDA arrays..." << std::endl;
    double *cuda_x;
    cudaMalloc((&cuda_x), sizeof(double) * N);
    // pointers to vectors on cpu
    double **cuda_y = (double **)malloc(sizeof(double *) * K / 8);
    for (size_t i = 0; i < K / 8; ++i)
    {
      cudaMalloc((void **)(&cuda_y[i]), sizeof(double) * N * 8);
    }
    double **cuda_results2 = (double **)malloc(sizeof(double *) * K / 8);
    for (size_t i = 0; i < K / 8; ++i)
    {
      cudaMalloc((void **)(&cuda_results2[i]), sizeof(double) * 8);
    }

    //
    // fill host arrays with values
    //
    std::fill(x, x + N, 1.0);
    for (size_t i = 0; i < K / 8; ++i)
    {
      for (size_t j = 0; j < N * 8; ++j)
      {
        y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
      }
    }
    // fill y_cpu
    for (size_t i = 0; i < K; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        y_cpu[i][j] = 1 + rand() / (1.1 * RAND_MAX);
      }
    }

    //
    // Reference calculation on CPU:
    //
    for (size_t i = 0; i < K; ++i)
    {
      results[i] = 0;
      for (size_t j = 0; j < N; ++j)
      {
        results[i] += x[j] * y_cpu[i][j];
      }
    }

    //
    // Copy data to GPU
    //
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    for (size_t i = 0; i < K / 8; ++i)
    {
      cudaMemcpy(cuda_y[i], y[i], sizeof(double) * N * 8, cudaMemcpyHostToDevice);
    }

    //
    // CUDA implementation
    //
    // repetitions
    for (int j = 0; j < reps; j++)
    {
      // wait for previous operations to finish, then start timings
      CUDA_ERRCHK(cudaDeviceSynchronize());
      timer.reset();

      for (int i = K / 8; i > 0; i--)
      {
        cuda_mdot_product<<<256, 256>>>(N, cuda_x, cuda_y[i - 1], cuda_results2[i - 1]);
      }

      CUDA_ERRCHK(cudaDeviceSynchronize());
      time_vec.push_back(timer.get());
    }
    std::sort(time_vec.begin(), time_vec.end());
    times.push_back(time_vec[reps / 2]);

    std::cout << "Copying data to CPU..." << std::endl;
    for (size_t i = 0; i < K / 8; ++i)
    {
      cudaMemcpy(results2[i], cuda_results2[i], sizeof(double) * 8, cudaMemcpyDeviceToHost);
    }

    //
    // Compare results
    //

    std::cout << "Copying results back to host..." << std::endl;

    for (size_t i = 0; i < K / 8; ++i)
    {
      for (size_t j = 0; j < 8; ++j)
      {
        std::cout << results[i * 8 + j] << " on CPU "
                  << results2[i][j] << " on GPU. Relative difference: "
                  << fabs(results[i * 8 + j] - results2[i][j]) / results[i * 8 + j] << std::endl;
      }
    }

    //
    // Clean up:
    // important: clean up inside of loop!
    std::cout << "Cleaning up..." << std::endl;
    for (int i = 0; i < K; ++i)
    {
      free(y_cpu[i]);
    }
    free(y_cpu);

    free(x);
    cudaFree(cuda_x);

    for (size_t i = 0; i < K / 8; ++i)
    {
      free(y[i]);
      cudaFree(cuda_y[i]);
      free(results2[i]);
      cudaFree(cuda_results2[i]);
    }
    free(y);
    free(cuda_y);

    free(results);
    free(results2);
    free(cuda_results2);
  }

  std::cout << "\ntime [s]:\n"
            << std::endl;
  for (const auto &value : times)
  {
    std::cout << value << "," << std::endl;
  }

  return 0;
}