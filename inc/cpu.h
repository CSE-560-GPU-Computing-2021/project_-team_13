#include "defines.h"

void applyGaussianFilter(Image &img_in, Image &img_out)
{
    int min_row, max_row, min_col, max_col, g_x, g_y, imageIndex;
    int gaussianSum;
    for (int row = 0; row < img_in.height; row++)
    {
        for (int col = 0; col < img_in.width; col++)
        {
            for (int channel = 0; channel < img_out.channels; channel++)
            {
                imageIndex = (row * img_in.width + col) * img_out.channels + channel;
                // printf("imageIndex: %d\n", imageIndex);
                gaussianSum = 0;
                min_row = fmax(row - GAUSSIAN_DIM / 2, 0);
                max_row = fmin(row + GAUSSIAN_DIM / 2 + 1, img_in.height);
                min_col = fmax(col - GAUSSIAN_DIM / 2, 0);
                max_col = fmin(col + GAUSSIAN_DIM / 2 + 1, img_out.width);

                g_x = 0;
                for (int offX = min_row; offX < max_row; offX++)
                {
                    g_y = 0;
                    for (int offY = min_col; offY < max_col; offY++)
                    {
                        gaussianSum += img_in.img[(offX * img_in.width + offY) * img_out.channels + channel] * GAUSSIAN[g_x][g_y];
                        g_y++;
                    }
                    g_x++;
                }
                img_out.img[imageIndex] = gaussianSum / 273;
            }
        }
    }
}

void convertToGrayScale(Image &img_out)
{
    int write = 0;
    for (int i = 0; i < img_out.size; i += 3)
        img_out.img[write++] = img_out.img[i] / 3 + img_out.img[i + 1] / 3 + img_out.img[i + 2] / 3;
    img_out.channels = 1;
    img_out.size = img_out.height * img_out.width;
}

void applySobelFilter(Image &img_out)
{
    int min_row, max_row, min_col, max_col, g_x, g_y, imageIndex;
    int sobelSum;
    unsigned char temp_img[img_out.size];
    for (int row = 0; row < img_out.height; row++)
    {
        for (int col = 0; col < img_out.width; col++)
        {

            imageIndex = (row * img_out.width + col) * img_out.channels;
            sobelSum = 0;
            min_row = fmax(row - SOBEL_DIM / 2, 0);
            max_row = fmin(row + SOBEL_DIM / 2 + 1, img_out.height);
            min_col = fmax(col - SOBEL_DIM / 2, 0);
            max_col = fmin(col + SOBEL_DIM / 2 + 1, img_out.width);
            // printf("row| min: %d max: %d\n", min_row, max_row);
            // printf("col| min: %d max: %d\n", min_col, max_col);
            g_x = 0;
            for (int offX = min_row; offX < max_row; offX++)
            {
                g_y = 0;
                for (int offY = min_col; offY < max_col; offY++)
                {
                    sobelSum += img_out.img[(offX * img_out.width + offY) * img_out.channels] * V_SOBEL[g_x][g_y];
                    // printf("%d ", sobelSum);
                    g_y++;
                }
                g_x++;
            }
            // printf("\n");
            temp_img[imageIndex] = sobelSum;
            // printf("%d %d\n", g_x, g_y);
            // if (sobelSum != 0)
            //     printf("sobelSum: %d\n", sobelSum);
        }
    }
    for (int i = 0; i < img_out.size; i++)
        img_out.img[i] = temp_img[i];
}

void preprocess(Image &img_in, Image &img_out)
{
    printf("Will preprocess here [CPU]\n");
    img_out.channels = img_in.channels;
    img_out.height = img_in.height;
    img_out.width = img_in.width;
    img_out.size = img_in.size;
    img_out.img = (unsigned char *)malloc(sizeof(unsigned char) * img_in.size);

    applyGaussianFilter(img_in, img_out);
    convertToGrayScale(img_out);
    applySobelFilter(img_out);
}