
// The following three defines are necessary to pick the correct OpenCL version on the machine:
#define VEXCL_HAVE_OPENCL_HPP
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>

#include "timer.hpp"
#include <vector>

int main()
{
    vex::Context ctx(vex::Filter::GPU && vex::Filter::DoublePrecision);

    std::cout << ctx << std::endl; // print list of selected devices

    Timer timer;

    for (int k = 0; k <= 7; k++)
    {

        int N = pow(10, k);

        std::vector<double> h_x(N, 1.0), h_y(N, 2.0);

        vex::vector<double> d_x(ctx, h_x);
        vex::vector<double> d_y(ctx, h_y);


        timer.reset();

        vex::vector<double> x_plus_y = d_x + d_y;
        vex::vector<double> x_minus_y = d_x - d_y;

        vex::Reductor<double, vex::SUM> sum(ctx);
        double res = sum(x_plus_y * x_minus_y);

        // std::cout << res << std::endl;

        std::cout << timer.get() << std::endl;
    }
    return 0;
}
