#include "defines.h"
#include <cuda_runtime.h>

extern void Preprocess_kernel(Image &img_in, Image &img_out);
extern void GetAverageIntensityOfRegions(dim3 grid, dim3 block, Image d_img, double *avgIntensity);
extern void ChanVeseCore(dim3 grid, dim3 block, Image &img, double *avgIntensity);


void ReadInputImage(Image &image_in, char *input_filename)
{
    image_in.img = stbi_load(input_filename, &image_in.width, &image_in.height, &image_in.channels, 3);
    image_in.size = image_in.width * image_in.height * image_in.channels;
    image_in.contour0 = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    image_in.contour = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    image_in.contourOld = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    printf("Height: %d\tWidth: %d\tChannels: %d\n", image_in.height, image_in.width, image_in.channels);
}

void PreProcess(Image &img_in, Image &img_out)
{
    Preprocess_kernel(img_in, img_out);
}

void initializeGPUImg(Image &h_img, Image &d_img)
{
    d_img.height = h_img.height;
    d_img.width = h_img.width;
    d_img.channels = h_img.channels;
    d_img.size = h_img.size;
    cudaMalloc(&d_img.img, sizeof(unsigned char) * d_img.size);
    cudaMalloc(&d_img.contour, sizeof(double) * d_img.size);
    cudaMalloc(&d_img.contour0, sizeof(double) * d_img.size);
    cudaMalloc(&d_img.contourOld, sizeof(double) * d_img.size);

    cudaMemcpy(d_img.img, h_img.img, sizeof(unsigned char) * d_img.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img.contour, h_img.contour, sizeof(double) * d_img.size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img.contour0, h_img.contour0, sizeof(double) * d_img.size, cudaMemcpyHostToDevice);
}

void FreeGPU(Image &d_img)
{
    cudaFree(d_img.img);
    cudaFree(d_img.contour);
    cudaFree(d_img.contour0);
    cudaFree(d_img.contourOld);
}

void printSum(Image img, double *arr)
{
    double s = 0;
    for (int i = 0; i < img.size; i++)
        s += arr[i];
    printf("sum: %f\n", s);
}

void RunChanVeseSegmentation(Image &img)
{
    img.copy(img.contour, img.contour0);

    Image d_img;
    initializeGPUImg(img, d_img);
    double *avgIntensity;
    cudaMalloc(&avgIntensity, sizeof(double) * 4);
    

    dim3 block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid = dim3(ceil((double)d_img.width / BLOCK_SIZE_X), ceil((double)d_img.height / BLOCK_SIZE_Y));

    for (int mainLoop = 0; mainLoop < ITERATIONS_BREAK; mainLoop++)
    {
        cudaMemset(avgIntensity, 0, sizeof(double) * 4);
        GetAverageIntensityOfRegions(grid, block, d_img, avgIntensity);
        // cudaDeviceSynchronize();
        for (int innerLoop = 0; innerLoop < ITERATIONS_BREAK; innerLoop++)
        {
            ChanVeseCore(grid, block, d_img, avgIntensity);
            // cudaDeviceSynchronize();
            cudaMemcpy(d_img.contour0, d_img.contour, sizeof(double) * d_img.size, cudaMemcpyDeviceToDevice);
        }
    }
    cudaMemcpy(img.contour, d_img.contour, sizeof(double) * d_img.size, cudaMemcpyDeviceToHost);
    FreeGPU(d_img);
}

void Paint(Image &img)
{
    for (int i = 0; i < img.height; i++)
        for (int j = 0; j < img.width; j++)
            if (img.contour[i * img.width + j] < 0)
                img.img[i * img.width + j] = 255;
            else
                img.img[i * img.width + j] = 0;
}

void DestroyImage(Image &img)
{
    //assert(false && "Not implemented");
}
