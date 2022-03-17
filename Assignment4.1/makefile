

aimosGPU: reduce-cuda.cu gpu-reduce-mpi.c clockcycle.h
	mpixlc -c -g gpu-reduce-mpi.c -o reduce-mpi.o
	nvcc -c -g -G gpu-reduce-cuda.cu -o reduce-cuda.o
	mpixlc -g reduce-cuda.o reduce-mpi.o -o gpu.out \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++

CPU: cpu-reduce.c clockcycle.h
	mpicc cpu-reduce.c -o cpu.out

localGPU: gpu-reduce-cuda.cu gpu-reduce-mpi.c clockcycle.h 
	mpicc -c -g gpu-reduce-mpi.c -o reduce-mpi.o
	nvcc -c -g -G gpu-reduce-cuda.cu -o reduce-cuda.o
	mpicc -g reduce-cuda.o reduce-mpi.o -o gpu.out \
-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++


