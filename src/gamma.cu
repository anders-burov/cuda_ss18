// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "gamma.cuh"

#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void computeGammaKernel(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc)
{
    // TODO (3.2) implement kernel for gamma correction
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (id_x < w && id_y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            imgOut[idx] = powf(imgIn[idx], gamma);
        }
    }
}


void computeGamma(float *imgOut, const float *imgIn, float gamma, size_t w, size_t h, size_t nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // TODO (3.1) compute gamma correction on CPU
    for (int c = 0; c < nc; c++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                size_t idx = c*w*h + j*w + i;
                imgOut[idx] = pow(imgIn[idx], gamma);
            }
        }
    }
}


void computeGammaCuda(float *imgOut, const float *imgIn, float gamma, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (3.2) specify suitable block size

    // TODO (3.2) implement computeGrid2D() in helper.cuh etc
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (3.2) execute gamma correction kernel
    computeGammaKernel <<<grid, block>>> (imgOut, imgIn, gamma, w, h, nc);

    // check for errors
    // TODO (3.2)
    CUDA_CHECK;
}
