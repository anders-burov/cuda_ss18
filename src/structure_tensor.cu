// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "structure_tensor.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

__device__
void computeEigenValuesOfMatrix2x2(float& lambda1, float& lambda2, const float& a, const float& b, const float& c, const float& d)
{
    float determinant = a*d - b*c;
    float trace = a + d;

    lambda1 = trace/2 + powf(trace*trace/4 - determinant, 0.5f);
    lambda2 = trace/2 - powf(trace*trace/4 - determinant, 0.5f);
}

__global__
void computeTensorOutputKernel(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // TODO (8.3) compute structure tensor output
    // alpha = 0.005
    // beta = 0.001

    if (nc != 3) return;
    if (x >= w || y >= h) return;

    // corner
    if (lmb2[y*w + x] >= lmb1[y*w + x] && lmb1[y*w + x] >= alpha)
    {
        imgOut[0*h*w + y*w + x] = 1.0f;
        imgOut[1*h*w + y*w + x] = 0.0f;
        imgOut[2*h*w + y*w + x] = 0.0f;
    }
    // edge
    else if (lmb1[y*w + x] <= beta && beta < alpha && alpha <= lmb2[y*w + x])
    {
        imgOut[0*h*w + y*w + x] = 1.0f;
        imgOut[1*h*w + y*w + x] = 1.0f;
        imgOut[2*h*w + y*w + x] = 0.0f;
    }
    // otherwise
    else
    {
        imgOut[0*h*w + y*w + x] = imgIn[0*h*w + y*w + x] * 0.5f;
        imgOut[1*h*w + y*w + x] = imgIn[1*h*w + y*w + x] * 0.5f;
        imgOut[2*h*w + y*w + x] = imgIn[2*h*w + y*w + x] * 0.5f;
    }
}


__global__
void computeDetectorKernel(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // TODO (8.1) compute eigenvalues
    float lambda1;
    float lambda2;
    computeEigenValuesOfMatrix2x2(lambda1, lambda2, tensor11[y*w + x], tensor12[y*w + x], tensor12[y*w + x], tensor22[y*w + x]);

    // TODO (8.2) implement detector
    if (lambda1 <= lambda2)
    {
        lmb1[y*w + x] = lambda1;
        lmb2[y*w + x] = lambda2;
    }
    else
    {
        lmb1[y*w + x] = lambda2;
        lmb2[y*w + x] = lambda1;
    }
}


__global__
void computeStructureTensorKernel(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    // TODO (7.3) compute structure tensor
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    tensor11[y*w + x] = 0;
    tensor12[y*w + x] = 0;
    tensor22[y*w + x] = 0;

    for (int z = 0; z < nc; z++)
    {
        tensor11[y*w + x] += dx[z*h*w + y*w + x]*dx[z*h*w + y*w + x];
        tensor12[y*w + x] += dx[z*h*w + y*w + x]*dy[z*h*w + y*w + x];
        tensor22[y*w + x] += dy[z*h*w + y*w + x]*dy[z*h*w + y*w + x];
    }
}


void computeTensorOutputCuda(float *imgOut, const float *lmb1, const float *lmb2, const float *imgIn, int w, int h, int nc, float alpha, float beta)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (8.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (8.3) execute kernel for computing tensor output
    computeTensorOutputKernel <<<grid, block>>> (imgOut, lmb1, lmb2, imgIn, w, h, nc, alpha, beta);

    // check for errors
    // TODO (8.3)
    CUDA_CHECK;
}


void computeDetectorCuda(float *lmb1, float *lmb2, const float *tensor11, const float *tensor12, const float *tensor22, int w, int h)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (8.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (8.2) execute kernel for detector
    computeDetectorKernel <<<grid, block>>> (lmb1, lmb2, tensor11, tensor12, tensor22, w, h);

    // check for errors
    // TODO (8.2)
    CUDA_CHECK;
}


void computeStructureTensorCuda(float *tensor11, float *tensor12, float *tensor22, const float *dx, const float *dy, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (7.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (7.3) execute structure tensor kernel
    computeStructureTensorKernel <<<grid, block>>> (tensor11, tensor12, tensor22, dx, dy, w, h, nc);

    // check for errors
    // TODO (7.3)
    CUDA_CHECK;
}
