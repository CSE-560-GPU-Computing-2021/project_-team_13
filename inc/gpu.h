#include "defines.h"

#include <cuda_runtime.h>

extern "C" void Preprocess_kernel(Image &img_in, Image &img_out);

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
    Preprocess_kernel(img_in , img_out);
}

void RunChanVeseSegmentation(Image &img)
{
    assert(false && "Not implemented");
}

void Paint(Image &img)
{
    assert(false && "Not implemented");
}

void DestroyImage(Image &img)
{
    assert(false && "Not implemented");
}
