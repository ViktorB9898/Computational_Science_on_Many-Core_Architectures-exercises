#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>


// result = (x, y)
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



/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
int main() {

  int N = 1000000;
  
  //
  // Allocate and initialize arrays on CPU
  //
  double *x = (double *)malloc(sizeof(double) * N);
  double *y = (double *)malloc(sizeof(double) * N);
  double alpha = 0;

  std::fill(x, x + N, 1);
  std::fill(y, y + N, 2);


  //
  // Allocate and initialize arrays on GPU
  //
  double *cuda_x;
  double *cuda_y;
  double *cuda_alpha;
  
  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double)));
  
  CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(cudaMemcpy(cuda_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));
  
  cuda_dot_product<<<256, 256>>>(N, cuda_x, cuda_y, cuda_alpha);
  
  CUDA_ERRCHK(cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));

  std::cout << "Result of dot product: " << alpha << std::endl;

  //
  // Clean up
  //
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_y));
  CUDA_ERRCHK(cudaFree(cuda_alpha));
  free(x);
  free(y);

  return EXIT_SUCCESS;
}

