// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "convolution.cuh"



int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{s|sigma|3.0|sigma}"
        "{r|repeats|1|number of computation repetitions}"
        "{c|cpu|false|compute on CPU}"
        "{m|mem|0|memory: 0=global, 1=shared, 2=texture}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    // compute on CPU
    bool cpu = cmd.get<bool>("cpu");
    std::cout << "mode: " << (cpu ? "CPU" : "GPU") << std::endl;
    float sigma = cmd.get<float>("sigma");
    std::cout << "sigma: " << sigma << std::endl;
    size_t memory = (size_t)cmd.get<int>("mem");
    if (memory == 1)
        std::cout << "memory: shared" << std::endl;
    else if (memory == 2)
        std::cout << "memory: texture" << std::endl;
    else
        std::cout << "memory: global" << std::endl;

    // init camera
    bool useCam = inputImage.empty();
    cv::VideoCapture camera;
    if (useCam && !openCamera(camera, 0))
    {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // read input frame
    cv::Mat mIn;
    if (useCam)
    {
        // read in first frame to get the dimensions
        camera >> mIn;
    }
    else
    {
        // load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
        mIn = cv::imread(inputImage.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    }
    // check
    if (mIn.empty())
    {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn, CV_32F);

    // init kernel
    int kradius = ceil(3 * sigma);    // TODO (5.1) calculate kernel radius using sigma
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 2*kradius+1;     // TODO (5.1) calculate kernel diameter from radius
    int kn = k_diameter*k_diameter;
    float *kernel = new float[kn];    // TODO (5.1) allocate array

    // TODO (5.1) implement createConvolutionKernel() in convolution.cu
    createConvolutionKernel(kernel, kradius, sigma);

    float *kernel2visualize = new float[kn];
    memcpy(kernel2visualize, kernel, kn* sizeof(float));
    float max_element_in_kernel = *std::max_element(kernel2visualize, kernel2visualize+kn);
    for (int i = 0; i < k_diameter*k_diameter; i++)
    {
        kernel2visualize[i] *= 1/max_element_in_kernel;
    }

    cv::Mat mKernel(k_diameter,k_diameter,CV_32FC1,kernel2visualize);

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    int n = w*h*nc;
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[n];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[n];

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_imgOut = NULL;
    float *d_kernel = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_imgIn, n* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_imgOut, n* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_kernel, kn* sizeof(float)); CUDA_CHECK;

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;
        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);

        if (cpu)
        {
            Timer timer;
            timer.start();
            // TODO (5.3) implement computeConvolution() in convolution.cu
            computeConvolution(imgOut, imgIn, kernel, kradius, w, h, nc);
            timer.end();
            float t = timer.get();
            std::cout << "time: " << t*1000 << " ms" << std::endl;
        }
        else
        {
            // upload to GPU

            // TODO copy all necessary arrays from host to device
            cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
            cudaMemcpy(d_kernel, kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;


            // if using constant memory
            // TODO (6.3) copy kernel from host to constant memory
            //std::cout << "using constant memory for kernel" << std::endl;

            Timer timer;
            timer.start();

            for(size_t i = 0; i < repeats; ++i)
            {
                if (memory == 1)
                {
                    // shared memory
                    // TODO (6.1) implement computeConvolutionSharedMemCuda() in convolution.cu
                    computeConvolutionSharedMemCuda(d_imgOut, d_imgIn, d_kernel, kradius, w, h, nc);
                }
                else if (memory == 2)
                {
                    // texture memory
                    // TODO (6.2) implement computeConvolutionTextureMemCuda() in convolution.cu
                    computeConvolutionTextureMemCuda(d_imgOut, d_imgIn, d_kernel, kradius, w, h, nc);
                }
                else
                {
                    // global memory
                    // TODO (5.4) implement computeConvolutionGlobalMemCuda() in convolution.cu
                    computeConvolutionGlobalMemCuda(d_imgOut, d_imgIn, d_kernel, kradius, w, h, nc);
                }
                cudaDeviceSynchronize();
            }

            timer.end();
            float t = timer.get()/repeats;
            std::cout << "time: " << t*1000 << " ms" << std::endl;

            // TODO copy all necessary arrays from device to host
            cudaMemcpy(imgOut, d_imgOut, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        }

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        // proceed similarly for other output images, e.g. the convolution kernel:
        if (!mKernel.empty())
            showImage("Kernel", mKernel, 100, 50);

        if (useCam)
        {
            // wait 30ms for key input
            if (cv::waitKey(30) >= 0)
            {
                mIn.release();
            }
            else
            {
                // retrieve next frame from camera
                camera >> mIn;
                // convert to float representation (opencv loads image values as single bytes by default)
                mIn.convertTo(mIn, CV_32F);
            }
        }
    }
    while (useCam && !mIn.empty());

    if (!useCam)
    {
        cv::waitKey(0);

        // save input and result
        cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
        cv::imwrite("image_result.png",mOut*255.f);
        cv::imwrite("image_kernel.png",mKernel*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_kernel); CUDA_CHECK;

    // TODO free memory of all host arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] kernel;
    delete[] kernel2visualize;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



