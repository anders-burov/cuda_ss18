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
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s*=2)
    {
        unsigned int idx = 2*s*tid;

        if (idx < blockDim.x) {
            sdata[idx] += sdata[idx + s];
        }

        __syncthreads();
    }

//    for(unsigned int s=1; s < blockDim.x; s *= 2) {
//        if (tid % (2*s) == 0) {
//        sdata[tid] += sdata[tid + s];
//        }
//        __syncthreads();
//    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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
    float *input = new float[n];

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
        cudaMemcpy(d_input, elemns, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        int length = n;

        for (;;)
        {
            // TODO (12.1) implement parallel reduction
            dim3 block(min(32, length), 1, 1);
            dim3 grid = computeGrid1D(block, length);
            int smBytes = block.x * sizeof(float);

            //std::cout << "grid size: " << grid.x << std::endl;

            reduceKernel <<<grid,block,smBytes>>> (d_input, d_output, length);
            //cudaDeviceSynchronize();
            std::swap(d_input, d_output);
            //cudaMemcpy(input, d_input, n* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

            length = grid.x;
            if (length <= 1) break;
        }
    }

    timer.end();
    float t = timer.get() / (float)repeats;  // elapsed time in seconds
    std::cout << "reduce0: " << t*1000 << " ms" << std::endl;

    // download result
    float *result = new float[1];
    // TODO (12.1) download result from d_output to result
    cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

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
    delete[] input;
}
