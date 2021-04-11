#include "common.h"
int GAUSSIAN[GAUSSIAN_DIM][GAUSSIAN_DIM] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};

double mysecond()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + ((double)tv.tv_usec / 1000000);
}
