// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "reduction.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#include "cublas_v2.h"

__global__ void reduceKernel(float *g_idata, float *g_odata, int n)
{
    // TODO (12.1) implement parallel reduction kernel
}


void runParallelReduction(int n, size_t repeats)
{
    // fill input array
    float *elemns = new float[n];
    int cpu = n;
    for(int i = 0; i < n; ++ i)
    {
        elemns[i] = 1;
    }

    // TODO (12.1) first implement parallel reduction (sum) on CPU (optional) and measure time

    // allocate arrays on GPU
    float *d_input = NULL;
    float *d_output = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_input, n *sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_output, n *sizeof(float)); CUDA_CHECK;

    Timer timer;
    timer.start();

    for(size_t k = 0; k < repeats; ++k)
    {
        // upload input to GPU
        // TODO (12.1) copy from elemns to d_input

        while(true)
        {
            // TODO (12.1) implement parallel reduction
            break;
        }
    }

    timer.end();
    float t = timer.get() / (float)repeats;  // elapsed time in seconds
    std::cout << "reduce0: " << t*1000 << " ms" << std::endl;

    // download result
    float *result = new float[1];
    // TODO (12.1) download result from d_output to result
    int sum = result[0];
    std::cout << "result reduce0: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;

    // create cublas handle
    cublasStatus_t stat;
    cublasHandle_t handle;
    // TODO (12.2) create handle using cublasCreate()
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed\n" << std::endl;
    }

    timer.start();
    for(size_t k = 0; k < repeats; ++k)
    {
        // upload input to GPU
        // TODO (12.2) copy from elemns to d_input
        cudaMemcpy(d_input, elemns, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        // TODO (12.2) call cublasSasum() and store output in result
        stat = cublasSasum(handle, n, d_input, 1, result);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "CUBLAS summation failed\n" << std::endl;
        }
    }
    timer.end();
    std::cout << "cublasSasum: " << timer.get() / (float)repeats << " ms" << std::endl;
    sum = result[0];
    std::cout << "result cublas: " << (int)sum << " (CPU=" << cpu << ")" << std::endl;

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS destruction failed\n" << std::endl;
    }

    cudaFree(d_input); CUDA_CHECK;
    cudaFree(d_output); CUDA_CHECK;

    delete[] elemns;
    delete[] result;
}
