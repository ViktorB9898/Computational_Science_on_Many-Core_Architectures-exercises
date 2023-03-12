
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "timer.hpp"

using Vector = std::vector<double>;


__global__ void addition(int n, int k, double *x, double *y, double *z)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for (size_t i = id; i < n-k; i += blockDim.x*gridDim.x){
    z[i+k] = x[i+k] + y[i+k];
    //printf("z: %g\n", z[i*k]);
  }
}

int main(void)
{

    double *x, *y, *z, *d_x, *d_y, *d_z;
    
    Vector bandwidth;
    Timer timer;

    //int N_min = 100000000;
    //int N_max = 100000000;
    int reps = 6; // has to be an even number
    
    int N = 100000000;

    // Allocate host memory and initialize
    x = (double*)malloc(N*sizeof(double));
    y = (double*)malloc(N*sizeof(double));
    z = (double*)malloc(N*sizeof(double));

    for (int i = 0; i < N; i++) {
        x[i] = i;
        y[i] = N-1-i;
        z[i] = 0;
    }
    
    // Allocate device memory and copy host data over
    cudaMalloc(&d_x, N*sizeof(double)); 
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_z, N*sizeof(double));
    
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

    for (int k = 1; k < 64; k++){
        //std::cout << "k = " << k << std::endl;

        Vector time_vec;
        
        // repetitions
        for (int j = 0; j < reps; j++){

            // MEASURING TIME FROM HERE
            cudaDeviceSynchronize();
            timer.reset();

            addition<<<256, 256>>>(N, k, d_x, d_y, d_z);

            cudaDeviceSynchronize();
            time_vec.push_back(timer.get());
            //std::cout << timer.get() << std::endl;
            
            // TO HERE
        }
        std::sort(time_vec.begin(), time_vec.end());
        bandwidth.push_back(3*(N-k)*sizeof(double)/time_vec[reps/2]*1e-9);
        //printf("Elapsed: %g\n", time);
    }
    cudaMemcpy(z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);

    printf("Gb/s:\n");
    for (const auto& value : bandwidth){
        std::cout << value << "," << std::endl;
    }

    
    return EXIT_SUCCESS;
}
    

