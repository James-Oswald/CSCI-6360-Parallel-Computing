EXECS=benchmark.out
MPICC?=mpicc

all: ${EXECS}

benchmark.out: benchmark.c
	${MPICC} -o benchmark.out benchmark.c

clean:
	rm -rf ${EXECS}

