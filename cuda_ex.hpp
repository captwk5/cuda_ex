#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include "cuda_runtime_api.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Cuda_Computing{
public:
    Cuda_Computing(){
        cudaGetDeviceCount(&m_gpu_count);
    }
    void test();
    int a = 3;

private:
    int m_gpu_count = 0;
    cudaDeviceProp prop;

    void cpuConvolution(int*** input, int*** output, int rows, int cols, int channels, vector<float> kernel);
};

template <class T>
class CudaArray{
public:
    explicit CudaArray() : start_(0), end_(0){

    }

    explicit CudaArray(size_t size){
        allocate(size);
    }

    ~CudaArray(){
        free();
    }

    size_t getSize() const{
        return end_ - start_;
    }

    // get data
    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    void set(const T* src, size_t size){
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess){
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    void get(T* dest, size_t size)
    {
        size_t min = std::min(size, getSize());
        cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

    int count = 0;

private:
    void allocate(size_t size){
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != cudaSuccess){
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        end_ = start_ + size;
    }

    void free(){
        if(start_ != 0){
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
};