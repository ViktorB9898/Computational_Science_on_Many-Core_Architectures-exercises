#include <iostream>
#include <vector>
#include "timer.hpp"

using Vector = std::vector<double>;

void malloc(){}

void measure_time(int N_min, int N_max, int repeats){

    Vector time_vec;
    double time_per_repeat;
    double time;
    for (int i = N_min; i < N_max; i*10){

        time = 0;
        for (int j = 0; j < repeats+1; j++){

            Timer timer;
            timer.reset();
            time += timer.get();

        }
    }
}

int main(){
    measure_time(10, 1000000, 10);

}