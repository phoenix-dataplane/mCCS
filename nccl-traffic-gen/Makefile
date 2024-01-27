.PHONY: all clean

CC = /usr/local/cuda-12.3/bin/nvcc
MPI_CFLAGS = -I/usr/lib/x86_64-linux-gnu/openmpi/include 
MPI_LDLIBS = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib64 -lmpi_cxx -lmpi
NCCL_CFLAGS = -I/nfs/yongji/nccl/build/include/
NCCL_LDLIBS = -L/nfs/yongji/nccl/build/lib/ -lnccl
LDLIBS += $(NCCL_LDLIBS) $(MPI_LDLIBS)
CFLAGS += $(NCCL_CFLAGS) $(MPI_CFLAGS) --std=c++17 -g -O3

APP = nccl-traffic-gen

all: $(APP)

$(APP): main.o
	$(CC) $(CFLAGS) -o $(APP) main.o $(LDLIBS)

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

clean:
	rm -f $(APP) *.o
