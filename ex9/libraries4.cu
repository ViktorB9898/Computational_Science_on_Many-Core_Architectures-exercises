
#include <vector>
#include <iostream>
#include "timer.hpp"

#define VIENNACL_WITH_CUDA

#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

int main()
{
  Timer timer;

  for (int k = 0; k <= 7; k++)
  {

    int N = pow(10, k);

    viennacl::vector<double> x = viennacl::scalar_vector<double>(N, 1.0);
    viennacl::vector<double> y = viennacl::scalar_vector<double>(N, 2.0);

    timer.reset();

    viennacl::vector<double> x_plus_y = x + y;
    viennacl::vector<double> x_minus_y = x - y;

    double res = viennacl::linalg::inner_prod(x_plus_y, x_minus_y);

    // std::cout << res << std::endl;

    std::cout << timer.get() << std::endl;
  }

  return EXIT_SUCCESS;
}
