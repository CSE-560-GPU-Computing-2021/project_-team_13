#include "defines.h"

int main(int argc, char *argv[])
{
    // Check if filename exists or not
    assert(argc > 1 && "Must specify input file");

    // Extracting input filename
    char *input_filename = argv[1];
    printf("Input filename: %s\n", input_filename);

    // Read input image_in
    Image image_in, image_out;
    image_in.img = stbi_load(input_filename, &image_in.width, &image_in.height, &image_in.channels, 3);
    image_in.size = image_in.width * image_in.height * image_in.channels;
    printf("Height: %d\tWidth: %d\tChannels: %d\n", image_in.height, image_in.width, image_in.channels);

    preprocess(image_in, image_out);

    // printf("%d\n", image_out.size);
    // for (int i = 0; i < image_out.size; i+=3)
    //     printf("%d %d %d -> %d %d %d\n", image_in.img[i], image_in.img[i+1], image_in.img[i+2], image_out.img[i], image_out.img[i+1], image_out.img[i+2]);
    
    stbi_write_png("output.png", image_out.width, image_out.height, image_out.channels, image_out.img, image_out.width * image_out.channels);
    free(image_out.img);
    // Free memory
    stbi_image_free(image_in.img);

    return 0;
}