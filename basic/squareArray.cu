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

__global__ void SquareArray(float *a)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    a[idx] *= a[idx];
}

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    


    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;

    float *d_a;
    cudaMalloc(&d_a, n * sizeof(float)); CUDA_CHECK;

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    
    dim3 block = dim3(n, 1, 1);
    dim3 grid = dim3(1, 1, 1);
    SquareArray <<<grid, block>>>(d_a);

    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaFree(d_a); CUDA_CHECK;
    
    // ###
    // ### TODO (2.1) Implement the "square array" operation on the GPU and store the result in "a"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"


    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
}



