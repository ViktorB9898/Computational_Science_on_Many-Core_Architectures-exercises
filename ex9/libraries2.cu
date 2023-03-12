#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <cstdlib>

#include <vector>
#include "timer.hpp"

int main(void)
{
  Timer timer;

  for (int k = 0; k <= 7; k++)
  {
    int N = pow(10, k);

    // x and y on host
    thrust::host_vector<int> h_x(N);
    thrust::fill(h_x.begin(), h_x.end(), 1);
    thrust::host_vector<int> h_y(N);
    thrust::fill(h_y.begin(), h_y.end(), 2);

    // transfer data to the device
    thrust::device_vector<int> d_x = h_x;
    thrust::device_vector<int> d_y = h_y;

    timer.reset();

    // x+y
    thrust::device_vector<float> x_plus_y(N);

    thrust::transform(
        d_x.begin(),
        d_x.end(),
        d_y.begin(),
        x_plus_y.begin(),
        thrust::plus<float>());

    // x-y
    thrust::device_vector<float> x_minus_y(N);

    thrust::transform(
        d_x.begin(),
        d_x.end(),
        d_y.begin(),
        x_minus_y.begin(),
        thrust::minus<float>());

    float res = thrust::inner_product(x_plus_y.begin(), x_plus_y.end(), x_minus_y.begin(), 0.f);

    // std::cout << res << std::endl;

    std::cout << timer.get() << std::endl;
  }

  return 0;
}
