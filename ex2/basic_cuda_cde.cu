
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "timer.hpp"

using Vector = std::vector<double>;


__global__ void addition(int n, double *x, double *y, double *z)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for (size_t i = id; i < n; i += blockDim.x*gridDim.x){
    z[i] = x[i] + y[i];
  }
}

int main(void)
{

    double *x, *y, *z, *d_x, *d_y, *d_z;
    
    Vector time_elapsed;
    Timer timer;

    int N_min = 100000000;
    int N_max = 100000000;
    int reps = 10; // has to be an even number
    
    for (int N = N_min; N <= N_max; N = N*5){
        std::cout << N << std::endl;

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

        Vector time_vec;
        
        // repetitions
        for (int j = 0; j < reps; j++){

            // MEASURING TIME FROM HERE
            cudaDeviceSynchronize();
            timer.reset();

            addition<<<256, 256>>>(N, d_x, d_y, d_z);

            cudaDeviceSynchronize();
            time_vec.push_back(timer.get());
            //std::cout << timer.get() << std::endl;
            
            // TO HERE
        }
        std::sort(time_vec.begin(), time_vec.end());
        time_elapsed.push_back(time_vec[reps/2]);
        //printf("Elapsed: %g\n", time);

        cudaMemcpy(z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

        // CHECK IF VALUES OF z ARE EQUAL (fast check on first two values)
        if (z[1] != z[2]){
            std::cout << "Values of z are not equal!!!!!" << std::endl;
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        free(x);
        free(y);
        free(z);

    }
    printf("Times:\n");
    for (const auto& value : time_elapsed){
        std::cout << value << std::endl;
    }


    
    return EXIT_SUCCESS;
}
    


