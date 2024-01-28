#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <ring_type> <job_color>    ring_type=goodring|badring, job_color=blue|red"
}

if [ $# -ne 2 ]; then
	usage
	exit 1
fi

ring_type=$1
color=$2

case $ring_type in
	goodring)
		cat > hostfile.$color.$ring_type <<EOF
danyang-02 slots=1
danyang-03 slots=1
danyang-01 slots=1
danyang-05 slots=1
EOF
		;;
	badring)
		cat > hostfile.$color.$ring_type <<EOF
danyang-02 slots=1
danyang-01 slots=1
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	*)
		echo "Error: ring_type should be either 'goodring' or 'badring', got $ring_type"
		usage
		exit 1
		;;
esac

case $color in
	blue)
		device_id=0
		tclass=106
		;;
	red)
		device_id=1
		tclass=66
		;;
	*)
		echo "Error: job_color should be either 'blue' or 'red', got $color"
		usage
		exit 1
		;;
esac

echo device_id = $device_id
echo Traffic class = $tclass

mpirun --hostfile hostfile.$color.$ring_type -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno1 \
	-x CUDA_VISIBLE_DEVICES=$device_id \
	-x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
	-x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=rdma0 \
	-x NCCL_MAX_NCHANNELS=2 -x NCCL_MIN_NCHANNELS=2 -x NCCL_IB_QPS_PER_CONNECTION=1 \
	-x NCCL_IB_TC=$tclass \
	-x NCCL_EPOCHS=20 \
		$WORKDIR/../build/all_reduce_perf --datatype=half -b 128M -e 128M
