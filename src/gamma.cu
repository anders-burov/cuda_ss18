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
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x < w && y < h)
    {
        int idx = z*h*w + y*w + x;
        imgOut[idx] = powf(imgIn[idx], gamma);
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
    for (size_t c = 0; c < nc; c++)
    {
        for (size_t j = 0; j < h; j++)
        {
            for (size_t i = 0; i < w; i++)
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
    dim3 block(8, 4, nc);     // TODO (3.2) specify suitable block size

    // TODO (3.2) implement computeGrid2D() in helper.cuh etc
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (3.2) execute gamma correction kernel
    computeGammaKernel <<<grid, block>>> (imgOut, imgIn, gamma, w, h, nc);

    // check for errors
    // TODO (3.2)
    CUDA_CHECK;
}
