// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "norm.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeNormKernel(float *imgOut, const float *u, int w, int h, int nc)
{
    // TODO (4.3) compute norm
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_x < w && id_y < h)
    {
        imgOut[id_y*w + id_x] = 0;
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            imgOut[idx] += u[idx]*u[idx];
        }
    }
}


void computeNormCuda(float *imgOut, const float *u, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (4.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.3) execute divergence kernel
    computeNormKernel <<<grid, block>>> (imgOut, u, w, h, nc);

    // check for errors
    // TODO (4.3)
    CUDA_CHECK;
}
