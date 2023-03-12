
    // specify use of OpenCL 1.2:
    #define CL_TARGET_OPENCL_VERSION  120
    #define CL_MINIMUM_OPENCL_VERSION 120

    #include <vector>
    #include <algorithm>
    #include <iostream>

    #include "timer.hpp"
     
    #include <boost/compute/algorithm/transform.hpp>
    #include <boost/compute/algorithm/inner_product.hpp>
    #include <boost/compute/container/vector.hpp>
    #include <boost/compute/functional/math.hpp>
     
    namespace compute = boost::compute;
     
    int main()
    {

        // get default device and setup context
        compute::device device = compute::system::default_device();
        compute::context context(device);
        compute::command_queue queue(context, device);

        Timer timer;

        for (int k = 0; k <=7; k++){

            int N = pow(10, k);

            

            // generate data on the host
            std::vector<float> h_x(N);
            std::fill(h_x.begin(), h_x.end(), 1);
            std::vector<float> h_y(N);
            std::fill(h_y.begin(), h_y.end(), 2);

            // create a vector on the device
            compute::vector<float> d_x (N, context);
            compute::vector<float> d_y (N, context);
        
            // transfer data from the host to the device
            compute::copy(h_x.begin(), h_x.end(), d_x.begin(), queue);
            compute::copy(h_y.begin(), h_y.end(), d_y.begin(), queue);
            
            timer.reset();
            
            // x+y
            compute::vector<float> x_plus_y(N, context);
            
            compute::transform(
                d_x.begin(),
                d_x.end(),
                d_y.begin(),
                x_plus_y.begin(),
                compute::plus<float>(),
                queue
            );
            
            // x-y
            compute::vector<float> x_minus_y(N, context);
            compute::transform(
                d_x.begin(),
                d_x.end(),
                d_y.begin(),
                x_minus_y.begin(),
                compute::minus<float>(),
                queue
            );
            
            float res = compute::inner_product(x_plus_y.begin(), x_plus_y.end(), x_minus_y.begin(), 0.f, queue);

            //std::cout << res << std::endl;

            std::cout << timer.get() << std::endl;
        }
     
        return 0;
    }
