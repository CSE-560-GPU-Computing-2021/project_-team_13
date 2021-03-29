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

    ReadInputImage(image_in, input_filename);

    PreProcess(image_in, image_out);

    RunChanVeseSegmentation(image_out);

    Paint(image_out);

    // printf("%d\n", image_out.size);
    // for (int i = 0; i < image_out.size; i+=3)
    //     printf("%d %d %d -> %d %d %d\n", image_in.img[i], image_in.img[i+1], image_in.img[i+2], image_out.img[i], image_out.img[i+1], image_out.img[i+2]);

    stbi_write_png("output.png", image_out.width, image_out.height, image_out.channels, image_out.img, image_out.width * image_out.channels);

    // Free memory
    stbi_image_free(image_in.img);
    image_in.img = NULL;
    DestroyImage(image_in);
    DestroyImage(image_out);

    return 0;
}