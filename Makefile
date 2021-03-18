all: cpu gpu

cpu: src/main.cpp inc/*
	g++ -DLAUNCH_CPU -Iinc/ src/main.cpp -o cpu

gpu: src/main.cpp inc/*
	nvcc -DLAUNCH_GPU -Iinc/ src/main.cpp -o gpu

run: cpu gpu
	./cpu images/saitama.png
	# ./gpu images/saitama.png

clean:
	rm -rf cpu gpu