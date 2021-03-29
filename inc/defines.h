#ifndef DEFINES_H
#define DEFINES_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "stbi_image.h"
#include "stbi_image_write.h"

typedef struct
{
    unsigned char *img = NULL;
    int height;
    int width;
    int channels;
    int size;
    int *contour = NULL;
} Image;

#define GAUSSIAN_DIM 5
#define SOBEL_DIM 3

#define EPSILON 0.000001
#define DT 0.1
#define H 1.0
#define lambda1 1.0
#define lambda2 1.0
#define MU 0.5
#define NU 0
#define P 1
#define ITERATIONS_BREAK 5
#define PI 3.14159265358979323846264338327950288
#define COLOR 100

int GAUSSIAN[GAUSSIAN_DIM][GAUSSIAN_DIM] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};

int V_SOBEL[SOBEL_DIM][SOBEL_DIM] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}};

void ReadInputImage(Image &image_in, char *input_filename);
void PreProcess(Image &img_in, Image &img_out);
void RunChanVeseSegmentation(Image &img);
void Paint(Image &img);

void DestroyImage(Image &img);

#ifdef LAUNCH_CPU
#include "cpu.h"
#endif

#ifdef LAUNCH_GPU
#include "gpu.h"
#endif

#endif