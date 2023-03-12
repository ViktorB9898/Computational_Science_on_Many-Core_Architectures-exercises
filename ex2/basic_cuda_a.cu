#include <iostream>
#include <vector>
#include "timer.hpp"

using Vector = std::vector<double>;

void measure_time(int N_min, int N_max, int repeats){

  Vector time_vec_malloc;
  Vector time_vec_free;

  double time_malloc, time_free;
  double *d_x;
  Timer timer;

  for (int i = N_min; i <= N_max; i = i*10){
    std::cout << i << std::endl;
    time_malloc = 0;
    time_free = 0;
    for (int j = 0; j < repeats+1; j++){
        cudaDeviceSynchronize();
        timer.reset();
        cudaMalloc(&d_x, i*sizeof(float));
        time_malloc += timer.get();
        //printf("Elapsed: %g\n", time_malloc);
        cudaDeviceSynchronize();
        timer.reset();
        cudaFree(d_x); 
        time_free += timer.get();
        //printf("Elapsed: %g\n", time);

    }
    time_vec_malloc.push_back(time_malloc/repeats);
    time_vec_free.push_back(time_free/repeats);
      

  }
  printf("malloc\n");
  for (const auto& value : time_vec_malloc){
    printf("%g\n", value);
  }
  printf("\nfree\n");
  for (const auto& value : time_vec_free){
    printf("%g\n", value);
  }
  
}

int main(){
  measure_time(10, 100000000, 10);

  return EXIT_SUCCESS;
}