// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "gradient.cuh"
#include "divergence.cuh"
#include "diffusion.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{n|iter|100|iterations}"
        "{e|epsilon|0.01|epsilon}"
        "{m|mode|0|diffusivity mode}"
        "{d|dt|0.001|dt}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    size_t iter = (size_t)cmd.get<int>("iter");
    std::cout << "iterations: " << iter << std::endl;
    float epsilon = cmd.get<float>("epsilon");
    std::cout << "epsilon: " << epsilon << std::endl;
    int mode = cmd.get<int>("mode");
    std::cout << "mode: " << mode << std::endl;
    float dt = cmd.get<float>("dt");
    if (dt == 0.0f)
        dt = 0.225f/funcDiffusivity(0, epsilon, mode);
    std::cout << "dt: " << dt << std::endl;

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

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    int n = nc*h*w;
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers
    cv::Mat mDiv(h,w,mIn.type());
    cv::Mat mV1(h,w,mIn.type());
    cv::Mat mV2(h,w,mIn.type());

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[n];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[n];
    float *div = new float[n];
    float *v1 = new float[n];
    float *v2 = new float[n];

    // allocate arrays on GPU
    float *d_imgIn = NULL;
    float *d_v1 = NULL;
    float *d_v2 = NULL;
    float *d_div = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_imgIn, n* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_v1, n* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_v2, n* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_div, n* sizeof(float)); CUDA_CHECK;

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);
        // upload to GPU
        // TODO copy from imgIn to d_imgIn
        cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        Timer timer;
        timer.start();
        for(size_t i = 0; i < iter; ++i)
        {
            // TODO (9.1) compute gradient of d_imgIn using computeGradientCuda() in gradient.cu
            computeGradientCuda(d_v1, d_v2, d_imgIn, w, h, nc);
            cudaDeviceSynchronize();

            // TODO (9.3) implement multDiffusivityCuda() in diffusion.cu
            multDiffusivityCuda(d_v1, d_v2, w, h, nc, epsilon, mode);
            cudaDeviceSynchronize();

            // TODO (9.4) compute divergence of d_v1, d_v2 using computeDivergenceCuda() in divergence.cu
            computeDivergenceCuda(d_div, d_v1, d_v2, w, h, nc);
            cudaDeviceSynchronize();

            // TODO (9.5) implement updateDiffusivityCuda() in diffusion.cu
            updateDiffusivityCuda(d_imgIn, d_div, w, h, nc, dt);
            cudaDeviceSynchronize();
        }
        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // download from GPU
        // TODO download from device arrays to host arrays
        cudaMemcpy(v1, d_v1, n* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(v2, d_v2, n* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(div, d_div, n* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(imgOut, d_imgIn, n* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mV1, v1);
        showImage("V1", mV1, 100+w+80, 100);
        convertLayeredToMat(mV2, v2);
        showImage("V2", mV2, 100+w+100, 100);
        convertLayeredToMat(mDiv, div);
        showImage("Div", mDiv, 100+w+60, 100);
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

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
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    cudaFree(d_imgIn);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_div);

    // TODO free memory of all host arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] div;
    delete[] v1;
    delete[] v2;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



