#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

# green or red: both take 2 gpus

usage() {
	echo "Usage: $0 <job_color>     ring_type=green|red"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

color=$1

case $color in
	blue)
		cat > hostfile.$color <<EOF
danyang-02 slots=2
danyang-01 slots=2
EOF
		;;
	red)
		cat > hostfile.$color <<EOF
danyang-03 slots=2
danyang-05 slots=2
EOF
		;;
	*)
		echo "Error: job_color should be either 'blue' or 'red', got $color"
		usage
		exit 1
		;;
esac

mpirun --hostfile hostfile.$color -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno1 \
	-x CUDA_VISIBLE_DEVICES=0,1 \
	-x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
	-x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=rdma0 \
	-x NCCL_MAX_NCHANNELS=2 -x NCCL_MIN_NCHANNELS=2 -x NCCL_IB_QPS_PER_CONNECTION=1 \
	-x NCCL_IB_TC=0 \
	-x NCCL_EPOCHS=10 \
		$WORKDIR/../build/all_reduce_perf --datatype=half -b 128M -e 128M
