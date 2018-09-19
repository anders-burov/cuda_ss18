// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "convolution.cuh"
#include "structure_tensor.cuh"


int main(int argc,char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{s|sigma|1.0|sigma}"
        "{a|alpha|0.005|alpha}"
        "{b|beta|0.001|beta}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    float sigma = cmd.get<float>("sigma");
    std::cout << "sigma: " << sigma << std::endl;
    float alpha = cmd.get<float>("alpha");
    std::cout << "alpha: " << alpha << std::endl;
    float beta = cmd.get<float>("beta");
    std::cout << "beta: " << beta << std::endl;

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
    int kradius = ceil(3 * sigma);    // TODO (7.1) calculate kernel radius using sigma
    std::cout << "kradius: " << kradius << std::endl;
    int k_diameter = 2*kradius+1;     // TODO (7.1) calculate kernel diameter from radius
    int kn = k_diameter*k_diameter;
    float *kernel = new float[k_diameter*k_diameter];    // TODO (7.1) allocate array
    createConvolutionKernel(kernel, kradius, sigma);

    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // ### Set the output image format
    //cv::Mat mOut(h,w,mIn.type());  // grayscale or color depending on input image, nc layers
    cv::Mat mOut(h,w,CV_32FC3);    // color, 3 layers
    cv::Mat mDx(h,w,CV_32FC3);    // color, 3 layers
    cv::Mat mDy(h,w,CV_32FC3);    // color, 3 layers
    cv::Mat mM11(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mM12(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mM22(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mLmb1(h,w,CV_32FC1);    // grayscale, 1 layer
    cv::Mat mLmb2(h,w,CV_32FC1);    // grayscale, 1 layer

    // ### Allocate arrays
    // allocate raw input image array
    float *imgIn = new float[nc*h*w];    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[nc*h*w];    // TODO allocate array
    float *dx = new float[nc*h*w];    // TODO allocate array
    float *dy = new float[nc*h*w];    // TODO allocate array
    float *t11 = new float[h*w];    // TODO allocate array
    float *t22 = new float[h*w];    // TODO allocate array
    float *t12 = new float[h*w];    // TODO allocate array
    float *lmb1 = new float[h*w];
    float *lmb2 = new float[h*w];

    // allocate arrays on GPU
    // kernel
    float *d_kernelGauss = NULL;
    // TODO alloc cuda memory for device arrays
    float kernelDx[9] = {-3/32.f, 0.f, 3/32.f, -10/32.f, 0.f, 10/32.f, -3/32.f, 0.f, 3/32.f};    // TODO (7.2) fill slide 19 lecture 1
    float kernelDy[9] = {-3/32.f, -10/32.f, -3/32.f, 0.f, 0.f, 0.f, 3/32.f, 10/32.f, 3/32.f};    // TODO (7.2) fill
    float *d_kernelDx = NULL;
    float *d_kernelDy = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_kernelGauss, kn* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_kernelDx, 9* sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_kernelDy, 9* sizeof(float)); CUDA_CHECK;

    // input
    float *d_imgIn = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_imgIn, nc*h*w* sizeof(float)); CUDA_CHECK;

    // output
    float *d_imgOut = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_imgOut, nc*h*w* sizeof(float)); CUDA_CHECK;

    // temp
    float *d_inSmooth = NULL;
    float *d_dx = NULL;
    float *d_dy = NULL;
    float *d_tensor11Nonsmooth = NULL;
    float *d_tensor12Nonsmooth = NULL;
    float *d_tensor22Nonsmooth = NULL;
    float *d_tensor11 = NULL;
    float *d_tensor12 = NULL;
    float *d_tensor22 = NULL;
    float *d_lmb1 = NULL;
    float *d_lmb2 = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_inSmooth, nc*h*w* sizeof(float)); CUDA_CHECK; // S
    cudaMalloc(&d_dx, nc*h*w* sizeof(float)); CUDA_CHECK; // dxS
    cudaMalloc(&d_dy, nc*h*w* sizeof(float)); CUDA_CHECK; // dyS
    cudaMalloc(&d_tensor11Nonsmooth, h*w* sizeof(float)); CUDA_CHECK; // m11
    cudaMalloc(&d_tensor12Nonsmooth, h*w* sizeof(float)); CUDA_CHECK; // m12
    cudaMalloc(&d_tensor22Nonsmooth, h*w* sizeof(float)); CUDA_CHECK; // m22
    cudaMalloc(&d_tensor11, h*w* sizeof(float)); CUDA_CHECK; // t11
    cudaMalloc(&d_tensor12, h*w* sizeof(float)); CUDA_CHECK; // t12
    cudaMalloc(&d_tensor22, h*w* sizeof(float)); CUDA_CHECK; // t22
    cudaMalloc(&d_lmb1, h*w* sizeof(float)); CUDA_CHECK; // lambda1
    cudaMalloc(&d_lmb2, h*w* sizeof(float)); CUDA_CHECK; // lambda2

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered (imgIn, mIn);

        // TODO (7.1) upload kernel to device
        cudaMemcpy(d_kernelGauss, kernel, kn * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        // TODO (7.2) upload kernelDx and kernelDy to device
        cudaMemcpy(d_kernelDx, kernelDx, 9 * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_kernelDy, kernelDy, 9 * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        // TODO upload input to device
        cudaMemcpy(d_imgIn, imgIn, nc*h*w * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        Timer timer;
        timer.start();

        // TODO (7.1) smooth imgIn using computeConvolutionGlobalMemCuda()
        computeConvolutionGlobalMemCuda(d_inSmooth, d_imgIn, d_kernelGauss, kradius, w, h, nc);
        cudaThreadSynchronize();

        // TODO (7.2) compute derivatives d_dx and d_dy using computeConvolutionGlobalMemCuda()
        computeConvolutionGlobalMemCuda(d_dx, d_inSmooth, d_kernelDx, 1, w, h, nc);
        computeConvolutionGlobalMemCuda(d_dy, d_inSmooth, d_kernelDy, 1, w, h, nc);
        cudaThreadSynchronize();

        // compute tensor
        // TODO (7.3) implement computeStructureTensorCuda() in structure_tensor.cu
        computeStructureTensorCuda(d_tensor11Nonsmooth, d_tensor12Nonsmooth, d_tensor22Nonsmooth, d_dx, d_dy, w, h, nc);  CUDA_CHECK;
        cudaThreadSynchronize();

        // blur tensor
        // TODO (7.4) blur non-smooth tensor images using computeConvolutionGlobalMemCuda()
        computeConvolutionGlobalMemCuda(d_tensor11, d_tensor11Nonsmooth, d_kernelGauss, kradius, w, h, 1);
        computeConvolutionGlobalMemCuda(d_tensor12, d_tensor12Nonsmooth, d_kernelGauss, kradius, w, h, 1);
        computeConvolutionGlobalMemCuda(d_tensor22, d_tensor22Nonsmooth, d_kernelGauss, kradius, w, h, 1);
        cudaThreadSynchronize();

        // compute detector
        // TODO (8.2) implement computeDetectorCuda() in structure_tensor.cu
        computeDetectorCuda(d_lmb1, d_lmb2, d_tensor11, d_tensor12, d_tensor22, w, h);
        cudaThreadSynchronize();

        // set output image
        // TODO (8.3) implement computeTensorOutputCuda() in structure_tensor.cu
        computeTensorOutputCuda(d_imgOut, d_lmb1, d_lmb2, d_imgIn, w, h, nc, alpha, beta);
        cudaThreadSynchronize();

        timer.end();
        float t = timer.get();
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // TODO copy all necessary arrays from device to host
        cudaMemcpy(dx, d_dx, nc*h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(dy, d_dy, nc*h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t11, d_tensor11Nonsmooth, h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t12, d_tensor12Nonsmooth, h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(t22, d_tensor22Nonsmooth, h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(lmb1, d_lmb1, h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(lmb2, d_lmb2, h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(imgOut, d_imgOut, nc*h*w* sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

        // show output image: first convert to interleaved opencv format from the layered raw array
        convertLayeredToMat(mOut, imgOut);
        showImage("Output", mOut, 100+w+40, 100);

        // TODO (7.5) visualize tensor images t11, t12, t22 (incl. scaling)
        convertLayeredToMat(mDx, dx);
        convertLayeredToMat(mDy, dy);
        showImage("Dx", mDx*10, 100, 100+h+40);
        showImage("Dy", mDy*10, 100+w+40, 100+h+40);

        memcpy(mM11.data, t11, h*w*sizeof(float));
        memcpy(mM12.data, t12, h*w*sizeof(float));
        memcpy(mM22.data, t22, h*w*sizeof(float));
        showImage("M11", mM11*10, 100, 100+h+40);
        showImage("M12", mM12*10, 100+w, 100+h+40);
        showImage("M22", mM22*10, 100+w*2, 100+h+40);

        memcpy(mLmb1.data, lmb1, h*w*sizeof(float));
        memcpy(mLmb2.data, lmb2, h*w*sizeof(float));
        showImage("lmb1", mLmb1*100, 100, 100+h+40);
        showImage("lmb2", mLmb2*100, 100+w+40, 100+h+40);

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
        //cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
        cv::imwrite("image_result.png", mOut*255.f);
        cv::imwrite("image_m11.png", mM11*10*255.f);
        cv::imwrite("image_m12.png", mM12*10*255.f);
        cv::imwrite("image_m22.png", mM22*10*255.f);
    }

    // ### Free allocated arrays
    // TODO free cuda memory of all device arrays
    cudaFree(d_kernelGauss); CUDA_CHECK;
    cudaFree(d_kernelDx); CUDA_CHECK;
    cudaFree(d_kernelDy); CUDA_CHECK;
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgOut); CUDA_CHECK;
    cudaFree(d_inSmooth); CUDA_CHECK;
    cudaFree(d_dx); CUDA_CHECK;
    cudaFree(d_dy); CUDA_CHECK;
    cudaFree(d_tensor11Nonsmooth); CUDA_CHECK;
    cudaFree(d_tensor12Nonsmooth); CUDA_CHECK;
    cudaFree(d_tensor22Nonsmooth); CUDA_CHECK;
    cudaFree(d_tensor11); CUDA_CHECK;
    cudaFree(d_tensor12); CUDA_CHECK;
    cudaFree(d_tensor22); CUDA_CHECK;
    cudaFree(d_lmb1); CUDA_CHECK;
    cudaFree(d_lmb2); CUDA_CHECK;

    // TODO free memory of all host arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] dx;
    delete[] dy;
    delete[] t11;
    delete[] t22;
    delete[] t12;
    delete[] lmb1;
    delete[] lmb2;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



