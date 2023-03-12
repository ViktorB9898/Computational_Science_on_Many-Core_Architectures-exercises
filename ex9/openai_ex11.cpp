/*
Given vectors x = (1, 1, . . . , 1) and y = (2, 2, . . . , 2) of size N , write a code to compute the dot product ⟨x + y, x − y⟩
with the Boost. Compute library and print the result.
*/

#include <boost/compute.hpp>
#include <iostream>

int main()
{   int k = 1;
    int N = 10**k; // size of vectors x and y

    // create the vector x
    boost::compute::vector<int> x(N, 1);

    // create the vector y
    boost::compute::vector<int> y(N, 2);

    // create the vector x + y
    boost::compute::vector<int> x_plus_y(N);
    boost::compute::transform(
        x.begin(), x.end(), y.begin(), x_plus_y.begin(),
        boost::compute::plus<int>()
    );

    // create the vector x - y
    boost::compute::vector<int> x_minus_y(N);
    boost::compute::transform(
        x.begin(), x.end(), y.begin(), x_minus_y.begin(),
        boost::compute::minus<int>()
    );

    // compute the dot product of x + y and x - y
    float result =
        boost::compute::inner_product(x_plus_y.begin(), x_plus_y.end(),
                                     x_minus_y.begin(), 0.f);

    // print the result
    std::cout << result << std::endl;

    return 0;
}