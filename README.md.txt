# CUDA Gaussian Blur Image Filter

## Project Description

This project implements a Gaussian blur filter for images using CUDA C++. The goal is to demonstrate the application of GPU parallelism to accelerate a common image processing task. It takes an input image, applies a Gaussian blur based on a specified sigma value, and saves the resulting blurred image. This project serves as a capstone demonstration for the "CUDA at Scale for the Enterprise" specialization, showcasing the ability to write, compile, and run CUDA code for a practical application.

The blur is implemented using a *separable* Gaussian filter approach. This means the 2D Gaussian convolution is broken down into two 1D convolutions: one horizontal pass followed by a vertical pass. This significantly reduces the computational cost compared to a direct 2D convolution, especially for larger blur radii.

## Features

*   **CUDA Kernel:** Implements Gaussian blur using GPU parallelism.
*   **Separable Filter:** Efficiently applies blur using horizontal and vertical passes.
*   **Image I/O:** Uses the `stb_image.h` and `stb_image_write.h` single-header libraries to load (PNG, JPG, BMP, etc.) and save (PNG) images, avoiding complex external dependencies.
*   **Command Line Interface:** Accepts input file, output file, and blur sigma as arguments.
*   **Error Handling:** Includes basic CUDA error checking.
*   **Timing:** Measures GPU kernel execution time using CUDA events.

## Dependencies

*   **NVIDIA CUDA Toolkit:** Version 10.x or later (tested with 11.x, 12.x). Ensure `nvcc` is in your PATH.
*   **C++ Compiler:** A C++ compiler compatible with `nvcc` (like GCC or Clang).
*   **Make:** Standard `make` utility to build the project using the provided `Makefile`.
*   **`stb_image.h` & `stb_image_write.h`:** Included in the `src/` directory. No separate installation needed.

## Building

Navigate to the project's root directory (`CUDA_GaussianBlur/`) in your terminal and run:

```bash
make
Use code with caution.
Markdown
This will compile the src/main.cu file and create an executable named blur_image in the current directory.
To clean up compiled files and output images:
make clean
Use code with caution.
Bash
Note: You might need to adjust the sm_XX architecture flag in the Makefile (-arch=sm_XX) to match your specific NVIDIA GPU's compute capability for optimal performance or compatibility. Common values include sm_60, sm_61, sm_70, sm_75, sm_80, sm_86. sm_75 is a reasonable default.
Running
Place your input image (e.g., a PNG file) in the data/ directory. A sample run.sh script is provided for convenience.
Using the script:
Make sure the script is executable: chmod +x run.sh
Then run:
./run.sh
Use code with caution.
Bash
This script assumes an input image named data/input.png, saves the output to output/blurred_image.png, and uses a sigma value of 5.0.
Running directly:
You can also run the executable directly from the command line, specifying the input file, output file, and sigma value:
./blur_image <path/to/input_image.png> <path/to/output_image.png> <sigma_value>
Use code with caution.
Bash
Example:
./blur_image data/input.png output/my_blurred_output.png 10.0
Use code with caution.
Bash
The program will:
Load the input image.
Allocate memory on the CPU and GPU.
Generate the 1D Gaussian kernel based on sigma.
Transfer data to the GPU.
Launch the CUDA kernels for horizontal and vertical blur passes.
Time the GPU execution.
Transfer the result back to the CPU.
Save the blurred image to the specified output path (the output/ directory will be created if it doesn't exist).
Clean up resources.
Project Structure
CUDA_GaussianBlur/
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── Makefile            # Build script for compiling the project
├── README.md           # This file
├── run.sh              # Example script to run the executable
├── data/               # Directory for input images
│   └── input.png       # Sample input image (add your own here)
├── output/             # Directory where output images are saved (created automatically)
│   └── blurred_image.png # Example output file
└── src/                # Directory for source code
    ├── main.cu         # Main CUDA C++ source file with host and device code
    ├── stb_image.h     # Single-header library for image loading
    └── stb_image_write.h # Single-header library for image writing
Use code with caution.
Algorithm Details
Gaussian Function: The blur uses weights derived from the Gaussian function: G(x) = exp(-x^2 / (2 * sigma^2)).
Kernel Generation: A 1D kernel is generated based on the specified sigma. The radius of the kernel is typically chosen as ceil(3 * sigma) to capture most of the Gaussian curve's energy. The kernel values are normalized so they sum to 1.
Separable Convolution: Instead of a 2D kernel, two 1D passes are used:
Horizontal Pass: Each output pixel is computed by convolving its corresponding row in the input image with the 1D Gaussian kernel horizontally. The result is stored in an intermediate buffer.
Vertical Pass: Each final output pixel is computed by convolving its corresponding column in the intermediate buffer with the same 1D Gaussian kernel vertically.
Parallelism: Each thread in the CUDA kernel is responsible for calculating the blurred value of one output pixel for a given pass (horizontal or vertical). Threads read neighboring pixel values (within the kernel radius) from global memory, apply the kernel weights, and write the final result for that pixel back to global memory.
Boundary Handling: Pixels near the image border require special handling as their neighborhood extends beyond the image bounds. This implementation uses a "clamp to edge" strategy: coordinates outside the image are clamped to the nearest valid edge coordinate.
Proof of Execution / Results
After running the program (e.g., via ./run.sh), you will find the blurred image in the output/ directory (e.g., output/blurred_image.png).
Input: data/input.png (Original image)
Output: output/blurred_image.png (Blurred version of the input)
You can compare these two images visually. The console output also provides information about the image dimensions, sigma used, and the GPU execution time, demonstrating that the CUDA kernels were executed.
(Self-assessment: Include sample input/output images in your repository/submission artifacts for grading).
Lessons Learned / Challenges
(This section is for you to fill in based on your experience building and running this project!)
Setting up CUDA: Initial challenges might involve configuring the CUDA environment, compiler flags (nvcc, -arch), and linking.
CUDA Programming Model: Understanding grid/block/thread hierarchy, kernel launch syntax (<<<...>>>), device memory management (cudaMalloc, cudaMemcpy, cudaFree).
Debugging: Debugging CUDA kernels can be tricky. Using printf within kernels (with caution) or NVIDIA's debugging tools (like cuda-gdb or Nsight Systems/Compute) is essential. Error checking (checkCudaErrors macro) is crucial.
Algorithm Mapping: Translating the sequential Gaussian blur algorithm into a parallel CUDA kernel, including handling thread indexing and boundary conditions correctly.
Optimization: Realizing the benefit of separable filters vs. 2D convolution. Exploring potential further optimizations like using shared memory to reduce global memory bandwidth usage (not implemented here for simplicity, but a good next step).
Image Libraries: Integrating simple libraries like stb_image makes image handling much easier than writing parsers from scratch.
Future Work / Potential Improvements
Shared Memory Optimization: Implement the blur passes using shared memory to cache input image tiles, reducing redundant global memory reads and potentially improving performance.
Texture Memory: Explore using CUDA texture memory, which can offer hardware-accelerated interpolation and boundary handling modes.
Support for Grayscale/Alpha: Modify the code to handle grayscale images (1 channel) or images with alpha channels (4 channels).
Different Filters: Extend the framework to support other image filters (e.g., Sobel edge detection, sharpening).
More Robust CLI: Use a library like getopt for more flexible command-line argument parsing.
Benchmarking: Perform more rigorous benchmarking comparing CPU vs. GPU implementations and different optimization levels.