all: cpu gpu

cpu: src/main.cpp inc/*
	g++ -g -DLAUNCH_CPU -O3 -Iinc/ src/main.cpp -o cpu

gpu: src/main.cpp inc/*
	nvcc -DLAUNCH_GPU -O3 -Iinc/ src/main.cpp -o gpu

run: cpu gpu
	./cpu images/saitama.png
	# ./gpu images/saitama.png

clean:
	rm -rf cpu gpu
