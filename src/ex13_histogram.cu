// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "histogram.cuh"

#define HIST_LENGTH 256

int main(int argc, char **argv)
{
    // parse command line parameters
    const char *params = {
        "{i|image| |input image}"
        "{b|bw|false|load input image as grayscale/black-white}"
        "{r|repeats|1|number of computation repetitions}"
        "{m|mode|0|mode}"
    };
    cv::CommandLineParser cmd(argc, argv, params);

    // input image
    std::string inputImage = cmd.get<std::string>("image");
    // number of computation repetitions to get a better run time measurement
    size_t repeats = (size_t)cmd.get<int>("repeats");
    // load the input image as grayscale
    bool gray = cmd.get<bool>("bw");
    int mode = cmd.get<int>("mode");

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
    std::cout << "image: " << w << " x " << h << std::endl;

    // initialize CUDA context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Set the output image format
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // allocate raw input image array
    float *imgIn = new float[n];    // TODO allocate array
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)


    // allocate arrays on GPU
    // input
    float *d_imgIn = NULL;
    // TODO alloc cuda memory for device arrays
    cudaMalloc(&d_imgIn, n *sizeof(float)); CUDA_CHECK;

    // histogram
    int nbins = 256;
    int *histogram = new int[nbins];    // TODO allocate array
    int *d_histogram = NULL;
    // TODO (13.1) alloc cuda memory for d_histogram
    cudaMalloc(&d_histogram, nbins*sizeof(int)); CUDA_CHECK;
    // TODO (13.1) reset values of d_histogram to 0
    memset(histogram, 0, nbins*sizeof(int));
    cudaMemcpy(d_histogram, histogram, nbins*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;
    //cudaMemset(&d_histogram, 0, nh*sizeof(int)); CUDA_CHECK;

    do
    {
        // convert range of each channel to [0,1]
        mIn /= 255.0f;

        // init raw input image array (and convert to layered)
        convertMatToLayered(imgIn, mIn);
        // upload to GPU
        // TODO upload input to device
        cudaMemcpy(d_imgIn, imgIn, n*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        Timer timer; timer.start();

        // Execute kernel
        for(size_t i = 0; i < repeats; ++i)
        {
            // TODO (13.1) implement computeHistogramCuda() in histogram.cu
            if (mode == 0) computeHistogramCuda(d_histogram, d_imgIn, nbins, w, h, nc);
            // TODO (13.3) implement computeHistogramCudaShared() in histogram.cu
            else if (mode == 1) computeHistogramCudaShared(d_histogram, d_imgIn, nbins, w, h, nc);
            else std::cerr << "mode not supported" << std::endl;

            cudaDeviceSynchronize();
        }

        float t = timer.get() / (float)repeats;  // elapsed time in seconds
        std::cout << "time: " << t*1000 << " ms" << std::endl;

        // Copy histogram back from GPU
        // TODO copy from d_histogram to histogram
        cudaMemcpy(histogram, d_histogram, nbins*sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;

        // show input image
        showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)
        // ### Display your own output images here as needed
        // TODO (13.2) show histogram using showHistogram256()
        showHistogram256("Histogram", histogram, 100 + w + 40, 100);

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
    }

    // free allocated arrays
    // TODO free cuda memory of all device arrays
    cudaFree(d_imgIn);
    cudaFree(d_histogram);

    // TODO free memory of all host arrays
    delete[] imgIn;
    delete[] histogram;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}



