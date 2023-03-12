#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

// result = (x, y)
__global__ void cuda_sums(int N, double *x, double *res)
{
    // clean res to wipe results from previous repetition 
    if (blockIdx.x * blockDim.x + threadIdx.x == 0){
        res[0] = 0;
        res[1] = 0;
        res[2] = 0;
        res[3] = 0;
    }
    // shared mem for each
    __shared__ double shared_sum[256];
    __shared__ double shared_abs_sum[256];
    __shared__ double shared_sq_sum[256];
    __shared__ double shared_zero;

    double entry = 0;
    double entry_abs = 0;
    double entry_sq = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        entry += x[i];
        entry_abs += abs(x[i]);
        entry_sq += x[i]*x[i];
        if (x[i] == 0){
            shared_zero++;
            //printf("zero_entries=%g\n", shared_zero);
            atomicAdd(&res[3], shared_zero);
        }
    }
    // write x to every shared memory
    shared_sum[threadIdx.x] = entry;
    shared_abs_sum[threadIdx.x] = entry_abs;
    shared_sq_sum[threadIdx.x] = entry_sq;

    for (int k = blockDim.x / 2; k > 0; k /= 2)
    {
        __syncthreads();
        if (threadIdx.x < k)
        {
            // sum
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + k];
            // abs sum
            shared_abs_sum[threadIdx.x] += shared_abs_sum[threadIdx.x + k];
            // square sum
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + k];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&res[0], shared_sum[0]);
        atomicAdd(&res[1], shared_abs_sum[0]);
        atomicAdd(&res[2], shared_sq_sum[0]);
    }
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
        // double *y = (double *)malloc(sizeof(double) * N);
        int res_size = 4; // size of result array
        double *result = (double *)malloc(sizeof(double) * res_size);

        std::fill(x, x + N, -2);
        std::fill(result, result + res_size, 0);
        // add a zero to x
        x[1] = 0;
        // std::fill(y, y + N, 2);

        // Allocate and initialize arrays on GPU

        double *cuda_x;
        //double *cuda_y;
        double *cuda_result;

        CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
        //CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
        CUDA_ERRCHK(cudaMalloc(&cuda_result, sizeof(double) * res_size));

        CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
        //CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaMemcpy(cuda_result, &result, sizeof(double) * res_size, cudaMemcpyHostToDevice));

        // repetitions
        for (int j = 0; j < reps; j++){
            // wait for previous operations to finish, then start timings
            CUDA_ERRCHK(cudaDeviceSynchronize());
            timer.reset();

            cuda_sums<<<256, 256>>>(N, cuda_x, cuda_result);

            CUDA_ERRCHK(cudaDeviceSynchronize());
            time_vec.push_back(timer.get());
        }
        std::sort(time_vec.begin(), time_vec.end());
        times.push_back(time_vec[reps/2]);
        bandwidth.push_back(N*sizeof(double)/time_vec[reps/2]*1e-9);

        CUDA_ERRCHK(cudaMemcpy(result, cuda_result, sizeof(double)*res_size, cudaMemcpyDeviceToHost));

        std::cout << "Result sum: " << result[0] << std::endl;
        std::cout << "Result abs sum: " << result[1] << std::endl;
        std::cout << "Result square sum: " << result[2] << std::endl;
        std::cout << "Result zero entries: " << result[3] << "\n" << std::endl;

        // Clean up

        CUDA_ERRCHK(cudaFree(cuda_x));
        //CUDA_ERRCHK(cudaFree(cuda_y));
        CUDA_ERRCHK(cudaFree(cuda_result));
        free(x);
        free(result);
        //free(y);
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
