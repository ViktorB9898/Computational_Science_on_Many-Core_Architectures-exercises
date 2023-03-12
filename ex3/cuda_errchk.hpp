#include <stdexcept>
#include <sstream>

inline void cuda_error_check_impl(cudaError error_code, const int line )
{
  if (cudaSuccess != error_code)
  {
    std::stringstream ss;
    ss << "(" << line << "): " << ": CUDA Runtime API error " << error_code << ": " << cudaGetErrorString( error_code ) << std::endl;
    throw std::runtime_error(ss.str());
  }
}

#define CUDA_ERRCHK(ARG) cuda_error_check_impl(ARG, __LINE__)
