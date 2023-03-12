#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void my_warp_reduction(int N, double *x, double *res) {

    if (blockIdx.x * blockDim.x + threadIdx.x == 0){
        res[0] = 0;
        res[1] = 0;
        res[2] = 0;
        res[3] = 0;
    }

    double sum = 0;
    double abs_sum = 0;
    double sq_sum = 0;
    double zeros = 0;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = id; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
        abs_sum += abs(x[i]);
        sq_sum += x[i]*x[i];
        if (x[i]==0) zeros++;
    }
    for (int i=16; i>0; i=i/2){
        sum += __shfl_down_sync(-1, sum, i);
        abs_sum += __shfl_down_sync(-1, abs_sum, i);
        sq_sum += __shfl_down_sync(-1, sq_sum, i);
        zeros += __shfl_down_sync(-1, zeros, i);
    } // thread 0 contains sum of all values

    if ((threadIdx.x & 31) == 0){
        atomicAdd(&res[0], sum);
        atomicAdd(&res[1], abs_sum);
        atomicAdd(&res[2], sq_sum);
        atomicAdd(&res[3], zeros);
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

            my_warp_reduction<<<256, 256>>>(N, cuda_x, cuda_result);
            //cuda_sums<<<256, 256>>>(N, cuda_x, cuda_result);

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
