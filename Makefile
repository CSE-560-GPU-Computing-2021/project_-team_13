all: cpu

cpu: src/main.cpp inc/*
	g++ -DLAUNCH_CPU -O3 -Iinc/ src/main.cpp -o cpu

gpu: src/main.cpp inc/*
	nvcc -DLAUNCH_GPU -O3 -Iinc/ src/main.cpp -o gpu

run: cpu
	./cpu images/test.pgm
	# ./gpu images/saitama.png

clean:
	rm -rf cpu gpu
