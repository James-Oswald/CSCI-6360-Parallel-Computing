
cpu.out: cpu-reduce-mpi.c clockcycle.h
	mpicc cpu-reduce-mpi.c -o cpu.out

localGPU: reduce-cuda.cu reduce-mpi.c clockcycle.h
	mpicc -g reduce-mpi.c -o reduce-mpi.o
	nvcc -g -G reduce-cuda.cu -o reduce-cuda.o
	mpicc -g reduce-cuda.o reduce-mpi.o -o gpu.out \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

aimosGPU: reduce-cuda.cu reduce-mpi.c clockcycle.h
	mpixlc -g reduce-mpi.c -o reduce-mpi.o
	nvcc -g -G reduce-cuda.cu -o reduce-cuda.o
	mpixlc -g reduce-cuda.o reduce-mpi.o -o gpu.out \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
