#include <iostream>

#include "cuda_runtime_api.h"
#include "cuda_ex.hpp"

#ifdef __NVCC__
    #define COMPILER 1
#else
    #define COMPILER 0
#endif

using namespace std;

int main(int argc, char** argv){

    if(COMPILER) {
        cout << "NVCC Compiler" << endl;

        Cuda_Computing cuda_c;
        cuda_c.test();
    }
    else{
        cout << "Not NVCC" << endl;
    }

    return 0;
}