#include "defines.h"

void ReadInputImage(Image &image_in, char *input_filename)
{
    image_in.img = stbi_load(input_filename, &image_in.width, &image_in.height, &image_in.channels, 3);
    image_in.size = image_in.width * image_in.height * image_in.channels;
    image_in.contour0 = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    image_in.contour = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    image_in.contourOld = (double *)malloc(sizeof(double) * image_in.width * image_in.height);
    printf("Height: %d\tWidth: %d\tChannels: %d\n", image_in.height, image_in.width, image_in.channels);
}

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
    img_out.img = (unsigned char *)realloc(img_out.img, sizeof(unsigned char) * img_out.size);
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
            g_x = 0;
            for (int offX = min_row; offX < max_row; offX++)
            {
                g_y = 0;
                for (int offY = min_col; offY < max_col; offY++)
                {
                    sobelSum += img_out.img[(offX * img_out.width + offY) * img_out.channels] * V_SOBEL[g_x][g_y];
                    g_y++;
                }
                g_x++;
            }
            temp_img[imageIndex] = sobelSum;
        }
    }
    for (int i = 0; i < img_out.size; i++)
        img_out.img[i] = temp_img[i];
}

void setInitBoundary(Image &img)
{

    double x;
    double y;
    double radius = (img.width < img.height) ? img.width / 2 : img.height / 2;
    radius *= radius;
    for (unsigned int i = 0; i < img.height; i++)
    {
        for (unsigned int j = 0; j < img.width; j++)
        {
            x = double(i) - img.height / 2.0;
            y = double(j) - img.width / 2.0;
            img.contour0[i * img.width + j] = 900.0 / (900.0 + x * x + y * y) - 0.5; //radius/(radius + x*x + y*y) - 0.5;
        }
    }
}

void PreProcess(Image &img_in, Image &img_out)
{
    img_out.channels = img_in.channels;
    img_out.height = img_in.height;
    img_out.width = img_in.width;
    img_out.size = img_in.size;
    img_out.contour0 = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
    img_out.contour = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
    img_out.contourOld = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
    img_out.img = (unsigned char *)malloc(sizeof(unsigned char) * img_out.size);
    memcpy(img_out.img, img_in.img, sizeof(unsigned char) * img_in.size);
    
    applyGaussianFilter(img_in, img_out);

    if(img_out.channels>1)
    	convertToGrayScale(img_out);
    setInitBoundary(img_out);
}

void GetAverageIntensityOfRegions(Image img, double &c1, double &c2)
{
    if (H == 0)
    {
        int i1 = 0, i2 = 0;
        for (int i = 0; i < img.height; i++)
        {
            for (int j = 0; j < img.width; j++)
            {
                if (img.contour[j + i * img.width] >= 0)
                {
                    c1++;
                    i1 += img.img[j + i * img.width];
                }
                else
                {
                    c2++;
                    i2 += img.img[j + i * img.width];
                }
            }
        }
        c1 = i1 / c1;
        c2 = i2 / c2;
    }
    else
    {
        double num1 = 0, den1 = 0, num2 = 0, den2 = 0, H_phi;
        for (int i = 0; i < img.height; i++)
        {
            for (int j = 0; j < img.width; j++)
            {
                H_phi = .5 * (1 + (2 / PI) * atan(img.contour[i * img.width + j] / H));
                num1 += ((double)img.img[i * img.width + j] * H_phi);
                den1 += H_phi;
                num2 += ((double)img.img[i * img.width + j]) * (1 - H_phi);
                den2 += 1 - H_phi;
            }
        }
        c1 = num1 / den1;
        c2 = num2 / den2;
    }
}

void GetConstantValues(Image &img, int i, int j, double &F1, double &F2, double &F3, double &F4, double &F, double &L, double &delPhi)
{
    double C1 = 1 / sqrt(EPSILON +
                         pow((img.contour[(i + 1) * img.width + j] - img.contour[i * img.width + j]), 2) +
                         pow((img.contour[i * img.width + j + 1] - img.contour[i * img.width + j - 1]), 2) / 4);

    double C2 = 1 / sqrt(EPSILON +
                         pow((img.contour[(i)*img.width + j] - img.contour[(i - 1) * img.width + j]), 2) +
                         pow((img.contour[(i - 1) * img.width + j + 1] - img.contour[(i - 1) * img.width + j - 1]), 2) / 4);

    double C3 = 1 / sqrt(EPSILON +
                         pow((img.contour[(i + 1) * img.width + j] - img.contour[(i - 1) * img.width + j]), 2) / 4.0 +
                         pow((img.contour[(i)*img.width + j + 1] - img.contour[(i)*img.width + j]), 2));

    double C4 = 1 / sqrt(EPSILON +
                         pow((img.contour[(i + 1) * img.width + j - 1] - img.contour[(i - 1) * img.width + j - 1]), 2) / 4.0 +
                         pow((img.contour[(i)*img.width + j] - img.contour[(i)*img.width + j - 1]), 2));

    delPhi = H / (PI * (H * H + (img.contour[i * img.width + j]) * (img.contour[i * img.width + j])));
    double Multiple = DT * delPhi * MU * (double(P) * pow(L, P - 1));
    F = H / (H + Multiple * (C1 + C2 + C3 + C4));
    Multiple = Multiple / (H + Multiple * (C1 + C2 + C3 + C4));
    F1 = Multiple * C1;
    F2 = Multiple * C2;
    F3 = Multiple * C3;
    F4 = Multiple * C4;
}

void Reinitialize(Image &img, int numIters)
{
    double a, b, c, d, x, G;

    int fstop = 0;
    int width = img.width;
    int height = img.height;
    // phi is img.contour

    for (int k = 0; k < numIters && fstop == 0; k++)
    {

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                img.contour[i * width + j] = img.contour[i * width + j];
            }
        }

        for (int i = 1; i < height - 1; i++)
        {
            for (int j = 1; j < width - 1; j++)
            {
                a = img.contour[i * width + j] - img.contour[(i - 1) * width + j];
                b = img.contour[(i + 1) * width + j] - img.contour[i * width + j];
                c = img.contour[i * width + j] - img.contour[i * width + (j - 1)];
                d = img.contour[i * width + (j + 1)] - img.contour[i * width + j];
                if (img.contour[i * width + j] > 0)
                {
                    G = sqrt(fmax(fmax(a, 0.0) * fmax(a, 0.0), fmin(b, 0.0) * fmin(b, 0.0)) + fmax(fmax(c, 0.0) * fmax(c, 0.0), fmin(d, 0.0) * fmin(d, 0.0))) - 1.0;
                }
                else if (img.contour[i * width + j] < 0)
                {
                    G = sqrt(fmax(fmin(a, 0.0) * fmin(a, 0.0), fmax(b, 0.0) * fmax(b, 0.0)) + fmax(fmin(c, 0.0) * fmin(c, 0.0), fmax(d, 0.0) * fmax(d, 0.0))) - 1.0;
                }
                else
                {
                    G = 0;
                }
                x = img.contour[i * width + j] >= 0 ? 1.0 : -1.0;
                img.contour[i * width + j] = img.contour[i * width + j] - 0.1 * x * G;
            }
        }
        // Check stopping condition
        double Q = 0.0;
        double M = 0.0;
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (abs(img.contour[i * width + j]) <= 1)
                {
                    M += 1;
                    int sun1 = img.contour[i * width + j];
                    int sun2 = img.contour[i * width + j];
                    Q += (sun1 - sun2) >= 0 ? sun1 - sun2 : sun2 - sun1;
                }
            }
        }
        if (M != 0)
        {
            Q = Q / ((double)M);
        }
        else
        {
            Q = 0.0;
        }

        if (Q < 0.1)
        {
            fstop = 1;
            //cout << "Stopping condition reached at " << k+1 << " iterations; Q = " << Q << endl;
        }
        else
        {
            //cout << "Iteration " << k << ", Q = " << Q << " > " << dt*h*h << endl;
        }
    }
}
double Sum(Image img);

void RunChanVeseSegmentation(Image &img)
{
    double c1, c2;
    double F, F1, F2, F3, F4, L = 1, delPhi;
    img.copy(img.contour, img.contour0);

    for (int mainLoop = 0; mainLoop < ITERATIONS_BREAK; mainLoop++)
    {
        img.copy(img.contourOld, img.contour);
        c1 = 0;
        c2 = 0;
        GetAverageIntensityOfRegions(img, c1, c2);

        for (int innerLoop = 0; innerLoop < ITERATIONS_BREAK; innerLoop++)
        {
            for (int i = 1; i < img.height - 1; i++)
            {
                for (int j = 1; j < img.width - 1; j++)
                {
                    GetConstantValues(img, i, j, F1, F2, F3, F4, F, L, delPhi);
                    double CurrPixel = img.contour[i * img.width + j] - DT * delPhi * (NU + lambda1 * pow(img.img[i * img.width + j] - c1, 2) - lambda2 * pow(img.img[i * img.width + j] - c2, 2));
                    img.contour[i * img.width + j] = F1 * img.contour[(i + 1) * img.width + j] +
                                                     F2 * img.contour[(i - 1) * img.width + j] +
                                                     F3 * img.contour[i * img.width + j + 1] +
                                                     F4 * img.contour[i * img.width + j - 1] + F * CurrPixel;
                }

            }

            for (int i = 0; i < img.height; i++)
            {
                img.contour[i * img.width] = img.contour[i * img.width + 1];
                img.contour[i * img.width + img.width - 1] = img.contour[i * img.width + img.width - 2];
                
            }

            for (int j = 0; j < img.width; j++)
            {
                img.contour[j] = img.contour[img.width + j];
                img.contour[(img.height - 1) * img.width + j] = img.contour[(img.height - 2) * img.width + j];
            }
            Reinitialize(img, 100);
        }
    }
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
    if (img.img != NULL)
        free(img.img);
    if (img.contour != NULL)
        free(img.contour);
    if (img.contour0 != NULL)
        free(img.contour0);
    if (img.contourOld != NULL)
        free(img.contourOld);
}
