// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "convolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// TODO (6.3) define constant memory for convolution kernel

// TODO (6.2) define texture for image


__global__
void computeConvolutionTextureMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.2) compute convolution using texture memory
}


__global__
void computeConvolutionSharedMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;

    int kdiameter = 2*kradius+1;

    if (id_x < w && id_y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + id_y*w + id_x;
            imgOut[idx] = 0;
            for (int v = -kradius; v <= kradius; v++)
            {
                for (int u = -kradius; u <= kradius; u++)
                {
                   imgOut[idx] += imgIn[c*w*h + max(min(id_y+v,h-1),0)*w + max(min(id_x+u,w-1),0)]*kernel[(v+kradius)*kdiameter+(u+kradius)];
                }
            }
        }
    }
}


__global__
void computeConvolutionGlobalMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (5.4) compute convolution using global memory
}


void createConvolutionKernel(float *kernel, int kradius, float sigma)
{
    // TODO (5.1) fill convolution kernel
    int kdiagonal = 2*kradius+1;
    for (int j = -kradius; j <= kradius; j++)
    {
        for (int i = -kradius; i <= kradius; i++)
        {
            kernel[(j+kradius)*kdiagonal+(i+kradius)] = expf(-(i*i+j*j)/(2*sigma*sigma))/(2*PI*sigma*sigma);
        }
    }
}


void computeConvolution(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    int kdiameter = 2*kradius+1;

    // TODO (5.3) compute convolution on CPU
    for (int c = 0; c < nc; c++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                int idx = c*w*h + j*w + i;
                imgOut[idx] = 0;
                for (int v = -kradius; v <= kradius; v++)
                {
                    for (int u = -kradius; u <= kradius; u++)
                    {
// Dirichlet Boundary
//                       if (j+v >= 0 && j+v < h && i+u >= 0 && i+u < w)
//                       {
//                           int conv_idx = c*w*h + (j+v)*w + (i+u);
//                           imgOut[idx] += imgIn[conv_idx]*kernel[(v+kradius)*kdiameter+(u+kradius)];
//                       }

// von Neuman Boundary
                         imgOut[idx] += imgIn[c*w*h + max(min(j+v,h-1),0)*w + max(min(i+u,w-1),0)]*kernel[(v+kradius)*kdiameter+(u+kradius)];
                    }
                }
            }
        }
    }
}


void computeConvolutionTextureMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (6.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // TODO (6.2) bind texture

    // run cuda kernel
    // TODO (6.2) execute kernel for convolution using global memory

    // TODO (6.2) unbind texture

    // check for errors
    // TODO (6.2)
}


void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (6.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // TODO (6.1) calculate shared memory size

    // run cuda kernel
    // TODO (6.1) execute kernel for convolution using global memory

    // check for errors
    // TODO (6.1)
}


void computeConvolutionGlobalMemCuda(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (5.4) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (5.4) execute kernel for convolution using global memory
    computeConvolutionGlobalMemKernel <<<grid, block>>> (imgOut, imgIn, kernel, kradius, w, h, nc);

    // check for errors
    // TODO (5.4)
    CUDA_CHECK;
}
