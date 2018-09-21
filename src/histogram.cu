// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "histogram.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeHistogramAtomicKernel(int *histogram, float *imgIn, int nbins, int w, int h, int nc)
{
    // TODO (13.1) update histogram using atomic operations
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = y*w + x;

    if (x >= w || y >= h) return;

    float intensity = 0.f;

    if (nc == 3) intensity = 0.3f*imgIn[0*h*w + idx] + 0.59f*imgIn[1*h*w + idx] + 0.11f*imgIn[2*h*w + idx];
    else if (nc == 1) intensity = imgIn[idx];
    else return;

    int bidx = 256*intensity;
    atomicAdd(&histogram[bidx], 1);
}


__global__
void computeHistogramAtomicSharedMemKernel(int *histogram, float *imgIn, int w, int h, int nc)
{
    // TODO (13.3) update histogram using atomic operations on shared memory
    extern __shared__ float shared_histogram[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= h*w) return;

    // zero-out the histogram
    shared_histogram[threadIdx.x] = 0;
    __syncthreads();

    // compute intensity and update the histogram
    float intensity = 0.f;
    if (nc == 3) intensity = 0.3f*imgIn[0*h*w + idx] + 0.59f*imgIn[1*h*w + idx] + 0.11f*imgIn[2*h*w + idx];
    else if (nc == 1) intensity = imgIn[idx];
    else return;

    int bidx = intensity*256;
    atomicAdd(&shared_histogram[bidx], 1);
    __syncthreads();

    // update the global histogram
    atomicAdd(&histogram[threadIdx.x], shared_histogram[threadIdx.x]);
}


void computeHistogramCuda(int *histogram, float *imgIn, int nbins, int w, int h, int nc)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (13.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);      // TODO (13.1) compute grid dimensions

    // run cuda kernel
    // TODO (13.1) execute kernel for histogram update using atomic operations
    computeHistogramAtomicKernel <<<grid, block>>> (histogram, imgIn, nbins, w, h, nc);

    // check for errors
    // TODO (13.1)
    CUDA_CHECK;
}

void computeHistogramCudaShared(int *histogram, float *imgIn, int nbins, int w, int h, int nc)
{
    if (!histogram)
    {
        std::cerr << "histogram not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(nbins, 1, 1);     // TODO (13.3) specify suitable block size
    dim3 grid = computeGrid1D(block, w*h);     // TODO (13.3) compute grid dimensions
    int smBytes = block.x*sizeof(int);

    // run cuda kernel
    // TODO (13.3) execute kernel for histogram update using atomic operations on shared memory
    computeHistogramAtomicSharedMemKernel <<<grid, block, smBytes>>> (histogram, imgIn, w, h, nc);

    // check for errors
    // TODO (13.3)
    CUDA_CHECK;
}
