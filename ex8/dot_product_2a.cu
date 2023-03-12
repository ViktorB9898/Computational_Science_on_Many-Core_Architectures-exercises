#include <iostream>
#include <vector>
#include "timer.hpp"
#include <algorithm>

using Vector = std::vector<double>;

const int threadsPerBlock = 256;

// KERNEL 2
__global__ void dot_partial(double *x, double *y, double *z_partial, int n)
{
	__shared__ double temp_array[threadsPerBlock]; // shared for both kernels
	double product = 0;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = id; i < n; i += blockDim.x * gridDim.x)
	{
		product += x[i] * y[i];
	}
	// add results as long as i < size of vectors
	temp_array[threadIdx.x] = product;

	// reduce in each block
	for (int i = blockDim.x / 2; i > 0; i = i / 2)
	{
		__syncthreads(); // sync threads in block

		// add elements
		if (threadIdx.x < i)
		{
			temp_array[threadIdx.x] += temp_array[threadIdx.x + i];
		}
	}
	// let only thread 0 write to memory
	if (threadIdx.x == 0)
	{
		z_partial[blockIdx.x] = temp_array[0];
	}
}
// KERNEL 1
__global__ void dot_sum(double *z_partial)
{
	for (int i = blockDim.x / 2; i > 0; i = i / 2)
	{
		__syncthreads();

		if (threadIdx.x < i)
			z_partial[threadIdx.x] += z_partial[threadIdx.x + i];
	}
}

int main(void)
{
	double *x, *y, *z, *d_x, *d_y, *d_z_partial;
	Vector times;
	Timer timer;

	Vector N_vec = {10000,50000,100000,500000,1000000,5000000,10000000,50000000};
	int repeats = 6;

	for (const auto &N : N_vec){
		Vector time_vec;
		std::cout << N << std::endl;
		// Allocate host memory and initialize
		z = (double *)malloc(threadsPerBlock * sizeof(double));
		x = (double *)malloc(N * sizeof(double));
		y = (double *)malloc(N * sizeof(double));

		for (int i = 0; i < N; i++)
		{
			x[i] = 1;
			y[i] = 1;
		}

		// Allocate device memory and copy host data over
		cudaMalloc(&d_x, N * sizeof(double));
		cudaMalloc(&d_y, N * sizeof(double));
		cudaMalloc(&d_z_partial, threadsPerBlock * sizeof(double));

		cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);
		// cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

		// better to measure time over whole loop than inside
		//  MEASURING TIME FROM HERE
		
		for (int n = 0; n < repeats; n++)
		{
			timer.reset();
			dot_partial<<<256, threadsPerBlock>>>(d_x, d_y, d_z_partial, N);

			cudaDeviceSynchronize();
			dot_sum<<<1, threadsPerBlock>>>(d_z_partial);

			cudaMemcpy(z, d_z_partial, threadsPerBlock * sizeof(double), cudaMemcpyDeviceToHost);
			time_vec.push_back(timer.get());
		}
		std::sort(time_vec.begin(), time_vec.end());
		times.push_back(time_vec[repeats/2]);
		// TO HERE

		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z_partial);
		free(x);
		free(y);
		free(z);
	}
	printf("Times:\n");
	for (const auto &value : times)
	{
		std::cout << value << std::endl;
	}

	return EXIT_SUCCESS;
}