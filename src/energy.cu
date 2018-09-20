// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "energy.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


__global__
void minimizeEnergySorStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    // TODO (11.3) implement SOR update step
}


__global__
void minimizeEnergyJacobiStepKernel(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda)
{
    // TODO (11.2) implement Jacobi update step
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if (x >= w || y >= h || z >= nc) return;

    float gr = (x+1 < w) ? diffusivity[y*w + x] : 0;
    float gl = (x > 0) ? diffusivity[y*w + (x-1)] : 0;
    float gu = (y+1 < h) ? diffusivity[y*w + x] : 0;
    float gd = (y > 0) ? diffusivity[(y-1)*w + x] : 0;

    float nominator = 2*imgData[z*h*w + y*w + x]
            + lambda*gr*uIn[z*h*w + y*w + min(x+1,w-1)]
            + lambda*gl*uIn[z*h*w + y*w + max(x-1,0)]
            + lambda*gu*uIn[z*h*w + min(y+1,h-1)*w + x]
            + lambda*gd*uIn[z*h*w + max(y-1,0)*w + x];

    float denominator = 2 + lambda*(gr+gl+gu+gd);

    uOut[z*h*w + y*w + x] = nominator/denominator;
}


__global__
void computeEnergyKernel(float *d_energy, float *a_in, float *d_imgData,
                        int w, int h, int nc, float lambda, float epsilon)
{
    // TODO (12.2) compute energy
}


void minimizeEnergySorStepCuda(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda, float sor_theta, int redOrBlack)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (11.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.3) execute kernel for SOR update step

    // check for errors
    // TODO (11.3)
}

void minimizeEnergyJacobiStepCuda(float *uOut, const float *uIn, const float *diffusivity, const float *imgData, int w, int h, int nc, float lambda)
{
    // calculate block and grid size
    dim3 block(32, 8, nc);     // TODO (11.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.2) execute kernel for Jacobi update step
    minimizeEnergyJacobiStepKernel <<<grid, block>>> (uOut, uIn, diffusivity, imgData, w, h, nc, lambda);

    // check for errors
    // TODO (11.2)
    CUDA_CHECK;
}

void computeEnergyCuda(float *d_energy, float *a_in, float *d_imgData,
                       int w, int h, int nc, float lambda, float epsilon)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (12.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (12.2) execute kernel for computing energy

    // check for errors
    // TODO (12.2)
}
