// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "divergence.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeDivergenceKernel(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // TODO (4.2) compute divergence
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h)
    {
        int idx = z*h*w + y*w + x;
        q[idx] = v1[idx] - v1[z*h*w + y*w + max(0,x-1)];
        q[idx] += v2[idx] - v2[z*h*w + max(0,y-1)*w + x];
    }
}


void computeDivergenceCuda(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, nc);     // TODO (4.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.2) execute divergence kernel
    computeDivergenceKernel <<<grid, block>>> (q, v1, v2, w, h, nc);

    // check for errors
    // TODO (4.2)
    CUDA_CHECK;
}
