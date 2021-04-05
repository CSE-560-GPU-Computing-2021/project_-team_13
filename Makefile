all: cpu gpu

cpu: src/main.cpp inc/*
	g++ -DLAUNCH_CPU -O3 -Iinc/ src/main.cpp -o cpu

gpu: src/main.cpp inc/*
	nvcc -DLAUNCH_GPU -O3 -Iinc/ src/main.cpp src/gpu_kernel.cu -o gpu

run: cpu gpu
	./cpu images/hand.jpeg
	# ./gpu images/saitama.png

clean:
	rm -rf cpu gpu
