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
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    // x-direction
    if (id_x > 1 && id_x < w && id_y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            int idx0 = c*h*w + id_y*w + id_x-1;
            q[idx] = v1[idx] - v1[idx0];
        }
    }

    // y-direction
    if (id_x < w && id_y > 1 && id_y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            int idx0 = c*h*w + (id_y-1)*w + id_x;
            q[idx] += v2[idx] - v2[idx0];
        }
    }
}


void computeDivergenceCuda(float *q, const float *v1, const float *v2, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (4.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.2) execute divergence kernel
    computeDivergenceKernel <<<grid, block>>> (q, v1, v2, w, h, nc);

    // check for errors
    // TODO (4.2)
    CUDA_CHECK;
}
