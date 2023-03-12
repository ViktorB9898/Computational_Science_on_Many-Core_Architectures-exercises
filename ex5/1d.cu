
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "timer.hpp"

using Vector = std::vector<double>;

__global__ void singleAdd(int n, double *x, double *y)
{
    for (unsigned int i=0; i < n; i++)
    {
        atomicAdd(&x[i],1);
    }
}

int main(void)
{
    double *x, *y, *d_x, *d_y;
    double elapsed;

    Vector add_per_second;

    Timer timer;
    int reps = 6; // has to be an even number

    Vector N_vec = {10000,100000,500000,1000000,5000000,10000000,50000000,100000000};
    for (const auto &N : N_vec)
    {
        Vector time_vec;

        x = (double*)malloc(N*sizeof(double));
        cudaMalloc(&d_x, N * sizeof(double));
        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

        y = (double*)malloc(N*sizeof(double));
        cudaMalloc(&d_y, N * sizeof(double));
        cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);

        // repetitions
        for (int j = 0; j < reps; j++)
        {

            // MEASURING TIME FROM HERE
            cudaDeviceSynchronize();
            timer.reset();

            singleAdd<<<1, 1>>>(N, d_x, d_y);

            cudaDeviceSynchronize();
            elapsed = timer.get();
            time_vec.push_back(elapsed);

            // TO HERE
        }
        std::sort(time_vec.begin(), time_vec.end());
        add_per_second.push_back(N/time_vec[reps / 2]);

        cudaFree(d_x);
        free(x);
    }

    printf("s:\n");
    for (const auto &value : add_per_second)
    {
        std::cout << value << "," << std::endl;
    }

    return EXIT_SUCCESS;
}
