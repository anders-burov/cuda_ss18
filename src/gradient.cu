// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gradient.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeGradientKernel(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // TODO (4.1) compute gradient in x-direction (u) and y-direction (v)
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    // x-direction
    if (x + 1 < w && y < h)
    {
        int idx = z*h*w + y*w + x;
        int idx1 = z*h*w + y*w + x+1;
        u[idx] = imgIn[idx1] - imgIn[idx];
    }

    // y-direction
    if (x < w && y+1 < h)
    {
        int idx = z*h*w + y*w + x;
        int idx1 = z*h*w + (y+1)*w + x;
        v[idx] = imgIn[idx1] - imgIn[idx];
    }
}


void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, nc);     // TODO (4.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.1) execute gradient kernel
    computeGradientKernel <<<grid, block>>> (u, v, imgIn, w, h, nc);

    // check for errors
    // TODO (4.1)
    CUDA_CHECK;
}
