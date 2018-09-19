// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "convolution.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"


// TODO (6.3) define constant memory for convolution kernel
#define KERNEL_MAX 41 // max diameter for constant kernel
__constant__ float constKernel[KERNEL_MAX*KERNEL_MAX];

// TODO (6.2) define texture for image
texture<float,2,cudaReadModeElementType> texRef;


__global__
void computeConvolutionTextureMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (6.2) compute convolution using texture memory
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    float val = tex2D(texRef, x+0.5f, y+0.5f);

    int kdiameter = 2*kradius+1;

    if (x < w && y < h)
    {
        for (int z = 0; z < nc; z++)
        {
            int idx = z*h*w + y*w + x;
            imgOut[idx] = 0;
            for (int v = -kradius; v <= kradius; v++)
            {
                for (int u = -kradius; u <= kradius; u++)
                {
                   imgOut[idx] += tex2D(texRef, x+u+0.5, max(min(y+v+0.5,h-0.5),0.5) + h*z) * kernel[(v+kradius)*kdiameter+(u+kradius)];
                }
            }
        }
    }
}


__global__
void computeConvolutionSharedMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc, int sm_x, int sm_y)
{
    // TODO (6.1) compute convolution using shared memory
    extern __shared__ float shared[];

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int block_size = blockDim.x*blockDim.y;
    int shared_size = sm_y*sm_x;

    int kdiameter = 2*kradius+1;

    for (int z = 0; z < nc; z++)
    {
        for (int i = threadIdx.y*blockDim.x+threadIdx.x; i < shared_size; i+=block_size)
        {
            // shared coordinates
            int u = i % sm_x;
            int v = i / sm_x;

            int glob_x = u - kradius + blockIdx.x*blockDim.x;
            int glob_y = v - kradius + blockIdx.y*blockDim.y;
            shared[i] = imgIn[z*h*w + max(min(glob_y,h-1),0)*w + max(min(glob_x,w-1),0)];
        }

        // set print array on
        // set logging file log.txt
        // set logging on
        // less log.txt | tr -d ',} {$=' | awk '{ if (NR % 26 == 0) { print $0"\0" } else { print $0 } }' | tr '\n' '\t' | tr '\0' '\n' | tail -c +2 | less > out.txt

        __syncthreads();

        if (x < w && y < h)
        {
            int idx = z*h*w + y*w + x;
            imgOut[idx] = 0;

            for (int j = 0; j < kdiameter; j++)
            {
                for (int i = 0; i < kdiameter; i++)
                {
                   //kernel from the global memory
                   //imgOut[idx] += shared[(threadIdx.y+j)*sm_x+(threadIdx.x+i)] * kernel[j*kdiameter+i];

                   //kernel from the constant memory
                   imgOut[idx] += shared[(threadIdx.y+j)*sm_x+(threadIdx.x+i)] * constKernel[j*KERNEL_MAX+i];
                }
            }
        }

        __syncthreads();
	}
}


__global__
void computeConvolutionGlobalMemKernel(float *imgOut, const float *imgIn, const float *kernel, int kradius, int w, int h, int nc)
{
    // TODO (5.4) compute convolution using global memory
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int kdiameter = 2*kradius+1;

    if (x < w && y < h)
    {
        for (int c = 0; c < nc; c++)
        {
            int idx = c*h*w + y*w + x;
            imgOut[idx] = 0;
            for (int v = -kradius; v <= kradius; v++)
            {
                for (int u = -kradius; u <= kradius; u++)
                {
                   imgOut[idx] += imgIn[c*w*h + max(min(y+v,h-1),0)*w + max(min(x+u,w-1),0)]*kernel[(v+kradius)*kdiameter+(u+kradius)];
                }
            }
        }
    }
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
    dim3 block(32, 32, 1);     // TODO (6.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    texRef.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRef.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRef.normalized = false; // access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)

    // TODO (6.2) bind texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texRef, imgIn, &desc, w, h*nc, w*sizeof(imgIn[0])); CUDA_CHECK;

    // run cuda kernel
    // TODO (6.2) execute kernel for convolution using global memory
    computeConvolutionTextureMemKernel <<<grid, block>>> (imgOut, imgIn, kernel, kradius, w, h, nc);
    CUDA_CHECK;

    // TODO (6.2) unbind texture
    cudaUnbindTexture(texRef);

    // check for errors
    // TODO (6.2)
    CUDA_CHECK;
}


void computeConvolutionSharedMemCuda(float *imgOut, const float *imgIn, const float *kernel, const float *kernel_cpu, int kradius, int w, int h, int nc)
{
    if (!imgOut || !imgIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    int kdiameter = 2*kradius+1;

    if (KERNEL_MAX < kdiameter)
    {
        std::cerr << "kernel diameter bigger than the allowed size!" << std::endl;
        return;
    }

    float *kernel_pitched = new float[KERNEL_MAX*KERNEL_MAX];
    for (int j = 0; j < kdiameter; j++)
    {
        for (int i = 0; i < kdiameter; i++)
        {
            kernel_pitched[j*KERNEL_MAX+i] = kernel_cpu[j*kdiameter+i];
        }
    }
    cudaMemcpyToSymbol(constKernel, kernel_pitched, KERNEL_MAX*KERNEL_MAX*sizeof(float)); CUDA_CHECK;

    // calculate block and grid size
    dim3 block(32, 32, 1);     // TODO (6.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // TODO (6.1) calculate shared memory size
    int sm_x = block.x + 2*kradius;
    int sm_y = block.y + 2*kradius;
    size_t smBytes = sm_x * sm_y * sizeof(float);

    // run cuda kernel
    // TODO (6.1) execute kernel for convolution using global memory
    computeConvolutionSharedMemKernel <<<grid, block, smBytes>>> (imgOut, imgIn, kernel, kradius, w, h, nc, sm_x, sm_y);

    // check for errors
    // TODO (6.1)
    CUDA_CHECK;

    delete[] kernel_pitched;
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
