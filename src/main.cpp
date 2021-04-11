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

    double t1, t2;

    t1 = mysecond();
    RunChanVeseSegmentation(image_out);
    t2 = mysecond();

    double time_taken = double(t2 - t1);
#ifdef LAUNCH_CPU
    printf("Time[CPU]: %f\n", time_taken);
#endif
#ifdef LAUNCH_GPU
    printf("Time[GPU]: %f\n", time_taken);
#endif

    Paint(image_out);

    stbi_write_png("output.png", image_out.width, image_out.height, image_out.channels, image_out.img, image_out.width * image_out.channels);

    // Free memory
    stbi_image_free(image_in.img);
    image_in.img = NULL;
    DestroyImage(image_in);
    DestroyImage(image_out);

    return 0;
}
