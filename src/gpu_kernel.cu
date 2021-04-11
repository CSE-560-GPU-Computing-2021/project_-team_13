
#include "common.h"


__global__ void GaussianFilter(unsigned char * img_in , unsigned char * img_out , int width , int height , int channels , int * gaussian_filter)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;


	if (col < width && row < height)
	{
		int gaussianSum , min_row , max_row , min_col , max_col , g_x , g_y , imageIndex;
		for (int channel = 0; channel < channels; channel++)
	    {
	        imageIndex = (row * width + col) * channels + channel;
	        gaussianSum = 0;
	        min_row = row - GAUSSIAN_DIM / 2 > 0 ? row - GAUSSIAN_DIM / 2 : 0;
	        max_row = (row + GAUSSIAN_DIM / 2 + 1) < height ? (row + GAUSSIAN_DIM / 2 + 1) :  height;
	        min_col = col - GAUSSIAN_DIM / 2 > 0 ? col - GAUSSIAN_DIM / 2 : 0;
	        max_col = (col + GAUSSIAN_DIM / 2 + 1) < width ? (col + GAUSSIAN_DIM / 2 + 1) : width;

	        g_x = 0;
	        for (int offX = min_row; offX < max_row; offX++)
	        {
	            g_y = 0;
	            for (int offY = min_col; offY < max_col; offY++)
	            {
	                gaussianSum += img_in[(offX * width + offY) * channels + channel] * gaussian_filter[g_x*GAUSSIAN_DIM + g_y];
	                g_y++;
	            }
	            g_x++;
	        }
		
	        img_out[imageIndex] = gaussianSum / 273;
	    }
	}
}

__global__ void RGB2GRAY(unsigned char * img_in , unsigned char * img_out , int width , int height , int channels)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;

	if (col < width && row < height)
	{
		int index = (row*width + col)*channels;
		img_out[row*width + col] = img_in[index] / 3 + img_in[index + 1] / 3 + img_in[index + 2] / 3;
	}
}

__global__ void InitContour(double * contour , int width , int height)
{
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int row = threadIdx.y + blockDim.y*blockIdx.y;

	if (col < width && row < height)
	{
		int x = double(row) - height / 2.0;
        int y = double(col) - width / 2.0;
        contour[row * width + col] = 900.0 / (900.0 + x * x + y * y) - 0.5; //radius/(radius + x*x + y*y) - 0.5;
    }
}


void Preprocess_kernel(Image &img_in, Image &img_out){
	img_out.channels = img_in.channels;
	img_out.height = img_in.height;
	img_out.width = img_in.width;
	img_out.size = img_in.size;
	img_out.contour0 = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
	img_out.contour = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
	img_out.contourOld = (double *)malloc(sizeof(double) * img_out.width * img_out.height);
	img_out.img = (unsigned char *)malloc(sizeof(unsigned char) * img_out.size);
	memcpy(img_out.img, img_in.img, sizeof(unsigned char) * img_in.size);

    int size = img_in.height * img_in.width * img_in.channels;
    unsigned char *d_img_in , *d_img_out , *d_img_flatten; double *d_img_contour;
    int *gaussian_filter, *d_gaussian_filter;
    
    gaussian_filter = (int * )malloc(sizeof(int) * GAUSSIAN_DIM * GAUSSIAN_DIM);
    for (int i=0;i<GAUSSIAN_DIM; i++){
		for(int j=0;j<GAUSSIAN_DIM; j++){
			gaussian_filter[i*GAUSSIAN_DIM + j] = GAUSSIAN[i][j];
		}
    }

    cudaMalloc((void**)&d_img_in, size*sizeof(unsigned char));
    cudaMalloc((void**)&d_img_out, size*sizeof(unsigned char));
    cudaMalloc((void**)&d_img_flatten, img_in.height * img_in.width * sizeof(unsigned char ));
    cudaMalloc((void**)&d_img_contour, img_in.height * img_in.width * sizeof(double));
    cudaMalloc((void**)&d_gaussian_filter, GAUSSIAN_DIM * GAUSSIAN_DIM * sizeof(int));
    
    cudaMemcpy(d_img_in, img_in.img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussian_filter, gaussian_filter, GAUSSIAN_DIM * GAUSSIAN_DIM * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_out, img_out.img, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	
    /*************** Kernel calls**************/
	dim3 grid , block;
	block.x = BLOCK_SIZE_X;
	block.y = BLOCK_SIZE_Y;
	grid.x = (img_in.width  % block.x==0) ? img_in.width  / block.x : img_in.width  / block.x+1;
	grid.y = (img_in.height % block.y==0) ? img_in.height / block.y : img_in.height / block.y+1;

	//kernel 1
	GaussianFilter <<< grid , block >>> (d_img_in, d_img_out , img_in.width , img_in.height , img_in.channels , d_gaussian_filter);

    // kernel 2  
    if(img_out.channels>1)
	  RGB2GRAY <<< grid , block >>>(d_img_out , d_img_flatten , img_in.width , img_in.height , img_in.channels);

    //kernel 3
    InitContour <<< grid , block >>>(d_img_contour , img_in.width , img_in.height);

    cudaDeviceSynchronize();

    img_out.channels = 1;
    img_out.size = img_out.height * img_out.width;
    
    cudaMemcpy(img_out.img, d_img_flatten, img_out.size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    img_out.img = (unsigned char *)realloc(img_out.img, sizeof(unsigned char) * img_out.size);
    cudaMemcpy(img_out.contour0, d_img_contour, img_out.size * sizeof(double), cudaMemcpyDeviceToHost);
	
    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_img_flatten);
    cudaFree(d_img_contour);
    cudaFree(d_gaussian_filter);
    free(gaussian_filter);
}

