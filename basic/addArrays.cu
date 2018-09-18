// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__global__ void AddArrays(float* a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];
    for(int i=0; i<n; i++)
    {
        a[i] = i;
        b[i] = (i%5)+1;
        c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) c[i] = a[i] + b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) c[i] = 0;
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_b, n * sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_c, n * sizeof(float)); CUDA_CHECK;

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_c, c, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    
    dim3 block = dim3(n, 1, 1);
    dim3 grid = dim3(1, 1, 1);
    AddArrays <<<grid, block>>> (d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_a); CUDA_CHECK;
    cudaFree(d_b); CUDA_CHECK;
    cudaFree(d_c); CUDA_CHECK;

    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << " + " << b[i] << " = " << c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    delete[] b;
    delete[] c;
}



