// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "diffusion.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include "helper.cuh"

#define EPS2 0.000001

__global__
void updateDiffusivityKernel(float *u, const float *d_div, int w, int h, int nc, float dt)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    // TODO (9.5) update diffusivity
    if (x >= w || y >= h || z >= nc) return;

    u[z*h*w + y*w + x] += dt * d_div[z*h*w + y*w + x];
}


__global__
void multDiffusivityKernel(float *v1, float *v2, int w, int h, int nc, float epsilon, int mode)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // TODO (9.3) multiply diffusivity
    if (x >= w || y >= h) return;

    float g = 0;
    for (int z = 0; z < nc; z++)
    {
        g += v1[z*h*w + y*w + x]*v1[z*h*w + y*w + x];
        g += v2[z*h*w + y*w + x]*v2[z*h*w + y*w + x];
    }
    g = funcDiffusivity(sqrtf(g), epsilon, mode);

    for (int z = 0; z < nc; z++)
    {
        v1[z*h*w + y*w + x] *= g;
        v2[z*h*w + y*w + x] *= g;
    }
}

__global__
void multDiffusivityAnisotropicKernel(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc)
{
    // TODO (10.2) multiply diffusivity (anisotropic)
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h) return;

    int idx = y*w + x;
    float temp_v1 = v1[idx];
    float temp_v2 = v2[idx];

    v1[idx] = g11[idx]*temp_v1 + g12[idx]*temp_v2;
    v2[idx] = g12[idx]*temp_v1 + g22[idx]*temp_v2;
}


__global__
void computeDiffusivityKernel(float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    // TODO (11.2) compute diffusivity
}

__device__
void computeEigenValuesAndVectorsOfMatrix2x2(float& lambda1, float& lambda2,
                                             float& v11, float& v12, float& v21, float& v22,
                                             const float& a, const float& b, const float& c, const float& d)
{
    float determinant = a*d - b*c;
    float trace = a + d;

    lambda1 = trace/2 + powf(trace*trace/4 - determinant, 0.5f);
    lambda2 = trace/2 - powf(trace*trace/4 - determinant, 0.5f);

    if (abs(c) > EPS2)
    {
        v11 = lambda1 - d; v12 = c;
        v21 = lambda2 - d; v22 = c;
    }
    else if (abs(b) > EPS2)
    {
        v11 = b; v12 = lambda1 - a;
        v21 = b; v22 = lambda2 - a;
    }
    else
    {
        v11 = 1; v12 = 0;
        v21 = 0; v22 = 1;
    }

//    if (lambda1 < lambda2)
//    {
//        float temp = lambda1;
//        lambda1 = lambda2;
//        lambda2 = temp;
//    }
}

__global__
void computeDiffusionTensorKernel(float *d_difftensor11, float *d_difftensor12, float *d_difftensor22,
                                  float *d_tensor11, float *d_tensor12, float *d_tensor22,
                                  float alpha, float C, int w, int h, int nc)
{
    // TODO (10.1) compute diffusion tensor    
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h) return;

    float lambda1,lambda2,v11,v12,v21,v22;
    int idx = y*w + x;

    computeEigenValuesAndVectorsOfMatrix2x2(lambda1, lambda2, v11, v12, v21, v22,
                                            d_tensor11[idx], d_tensor12[idx], d_tensor12[idx], d_tensor22[idx]);

    float mu1 = alpha;
    float mu2 = (abs(lambda1 - lambda2) < EPS2) ? alpha : alpha + (1-alpha)*expf(-C/powf(lambda1-lambda2,2));

    d_difftensor11[idx] = mu1*v11*v11 + mu2*v21*v21;
    d_difftensor12[idx] = mu1*v11*v12 + mu2*v21*v22;
    d_difftensor22[idx] = mu1*v12*v12 + mu2*v22*v22;
}


void updateDiffusivityCuda(float *u, const float *d_div, int w, int h, int nc, float dt)
{
    // calculate block and grid size
    dim3 block(32, 8, nc);     // TODO (9.5) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (9.5) execute kernel for updating diffusivity
    updateDiffusivityKernel <<<grid, block>>> (u, d_div, w, h, nc, dt);

    // check for errors
    // TODO (9.5)
    CUDA_CHECK;
}


void multDiffusivityCuda(float *v1, float *v2, int w, int h, int nc, float epsilon, int mode)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (9.3) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (9.3) execute kernel for multiplying diffusivity
    multDiffusivityKernel <<<grid, block>>> (v1, v2, w, h, nc, epsilon, mode);

    // check for errors
    // TODO (9.3)
    CUDA_CHECK;
}


void multDiffusivityAnisotropicCuda(float *v1, float *v2, float *g11, float *g12, float *g22, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (10.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (10.2) execute kernel for multiplying diffusivity (anisotropic)
    multDiffusivityAnisotropicKernel <<<grid, block>>> (v1, v2, g11, g12, g22, w, h, nc);

    // check for errors
    // TODO (10.2)
    CUDA_CHECK;
}


void computeDiffusivityCuda(float *diffusivity, const float *u, int w, int h, int nc, float epsilon)
{
    // calculate block and grid size
    dim3 block(0, 0, 0);     // TODO (11.2) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (11.2) execute kernel for computing diffusivity

    // check for errors
    // TODO (11.2)
}


void computeDiffusionTensorCuda(float *d_difftensor11, float *d_difftensor12, float *d_difftensor22,
                                float *d_tensor11, float *d_tensor12, float *d_tensor22,
                                float alpha, float C, int w, int h, int nc)
{
    // calculate block and grid size
    dim3 block(32, 8, 1);     // TODO (10.1) specify suitable block size
    dim3 grid = computeGrid2D(block, w, h);

    // run cuda kernel
    // TODO (10.1) execute kernel for computing diffusion tensor
    computeDiffusionTensorKernel <<<grid, block>>> (d_difftensor11, d_difftensor12, d_difftensor22,
                                      d_tensor11, d_tensor12, d_tensor22,
                                      alpha, C, w, h, nc);

    // check for errors
    // TODO (10.1)
    CUDA_CHECK;
}
