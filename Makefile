all: cpu gpu

cpu: src/main.cpp inc/* src/common.cpp
	g++ -DLAUNCH_CPU -O3 -Iinc/ src/main.cpp src/common.cpp  -o cpu

gpu: src/main.cpp inc/* src/gpu_kernel.cu src/common.cpp
	nvcc -DLAUNCH_GPU -arch=sm_60 -O3 -Iinc/ src/main.cpp src/gpu_kernel.cu src/common.cpp -o gpu

run: cpu gpu
	./cpu images/saitama.jpg
	./gpu images/saitama.jpg

clean:
	rm -rf cpu gpu
