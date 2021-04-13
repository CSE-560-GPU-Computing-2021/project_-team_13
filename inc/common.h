#ifndef COMMONS_H_FILE
#define COMMONS_H_FILE
/*****************GPU******************/
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
/*****************GPU******************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

typedef struct
{
    int height;
    int width;
    int channels;
    int size;
    unsigned char *img = NULL;
    double *contour = NULL;
    double *contour0 = NULL;
    double *contourOld = NULL;

    void copy(double *to, double *from)
    {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                to[i * width + j] = from[i * width + j];
    }
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


extern int GAUSSIAN[GAUSSIAN_DIM][GAUSSIAN_DIM];
extern double mysecond();
#endif
