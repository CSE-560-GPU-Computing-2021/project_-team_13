#ifndef DEFINES_H
#define DEFINES_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "common.h"
#include "stbi_image.h"
#include "stbi_image_write.h"

void ReadInputImage(Image &image_in, char *input_filename);
void PreProcess(Image &img_in, Image &img_out);
void RunChanVeseSegmentation(Image &img);
void Paint(Image &img);

void DestroyImage(Image &img);

int V_SOBEL[SOBEL_DIM][SOBEL_DIM] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};



#ifdef LAUNCH_CPU
#include "cpu.h"
#endif

#ifdef LAUNCH_GPU
#include "gpu.h"
#endif

#endif
