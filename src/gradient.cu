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
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    // x-direction
    if (id_x + 1 < w && id_y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            int idx1 = c*h*w + id_y*w + id_x+1;
            u[idx] = imgIn[idx1] - imgIn[idx];
        }
    }

    // y-direction
    if (id_x < w && id_y+1 < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            int idx1 = c*h*w + (id_y+1)*w + id_x;
            v[idx] = imgIn[idx1] - imgIn[idx];
        }
    }
}


void computeGradientCuda(float *u, float *v, const float *imgIn, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (4.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (4.1) execute gradient kernel
    computeGradientKernel <<<grid, block>>> (u, v, imgIn, w, h, nc);

    // check for errors
    // TODO (4.1)
    CUDA_CHECK;
}
